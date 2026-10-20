"""Building a game by hand, and building an opponent for it.

Both halves of this file exist for the same reason and are the same shape. A ruleset is
declarative data and so is a heuristic player -- schema-validated, content-hashed, no
generated code -- which is exactly what makes an *interactive editor* for either of them
possible at all. Nothing here can widen what the DSL can express; it can only fill in a
form the schema already accepts, and then ask the same validator the command line asks.

Three rules the browser does not get to bend:

**Validation is `spec/schema.py`'s and the gate is `loop/gates.py`'s.** A game built here
is refused for the same reasons and in the same words as one a model proposed. The editor
knows nothing about what makes a spec legal.

**A new game is a candidate, and nothing more.** It is written into a design-run archive
under ``runs/``, which git ignores, so the ladder out of it is the one every candidate
climbs: archive it, then promote it. Building a game must not be a way to put something in
the calibration corpus without deciding to.

**A hand-built opponent is kept apart from the measured ones.** ``heuristics/<hash>.json``
is the committed set the heuristic pass measures, and `heuristic_headroom` is a median
over the set -- so an opponent somebody invented while playing must not land in it. It
goes to ``heuristics/custom/``, which is a third store beside ``model/`` and is read only
by the picker.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from chess_variant_sandbox.agents.evaluation import Evaluator, PieceValues
from chess_variant_sandbox.agents.features import GLOSSES, MAX_ABS_WEIGHT, VOCABULARY
from chess_variant_sandbox.agents.heuristics import (
    STORE_DIR,
    Heuristic,
    HeuristicError,
    HeuristicSet,
    other_spelling,
    store_path,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.coming import coming_phases
from chess_variant_sandbox.engine.compile import HexCopy
from chess_variant_sandbox.loop.archive import Archive, Entry
from chess_variant_sandbox.loop.gates import STALLED
from chess_variant_sandbox.spec import (
    SCHEMA_VERSION,
    SpecError,
    reference,
    spec_hash,
    validate_spec,
)
from chess_variant_sandbox.spec.describe import (
    RULE_WORDS,
    describe_changes,
    describe_pieces,
    describe_rules,
    describe_setup,
    move_sentence,
    plain_move_sentence,
)
from chess_variant_sandbox.spec.plain import field_words
from chess_variant_sandbox.spec.plain import plain_problem as _spec_plain_problem
from chess_variant_sandbox.spec.schema import (
    HEX_SYMMETRIES,
    MAX_PIECE_TYPES,
    MAX_PORTAL_ENDS,
    MAX_PORTAL_GROUPS,
    PORTAL_ENDS,
    SQUARE_SYMMETRIES,
    portal_end_ids,
    portal_groups,
    portal_map_shape,
)
from chess_variant_sandbox.spec.transform import settle_schema_version
from chess_variant_sandbox.viz.pieces import icons_for
from chess_variant_sandbox.viz.walks import walk_payload
from chess_variant_sandbox.viz.wrapped import (
    fold_transit,
    portal_transit,
    walls_payload,
)
from chess_variant_sandbox.web.card_rules import preview as _preview

# Where a game built in the browser goes. One directory, shared by everything designed
# by hand, because it is an archive in the same sense a design run's is -- an append-only
# record of what was tried -- and `corpus.find` looks for candidates by globbing
# `runs/*/lineage.jsonl`. A directory per game would work and would make the library's
# scan of design runs grow without bound.
DESIGNED_RUN = "designed"

# The sub-store for opponents built here. Beside `heuristics/model/`, and for the reason
# that one exists: `heuristic_headroom` is a median over a set, so what is in the set
# changes the measurement.
CUSTOM_HEURISTICS = "custom"

# What a piece-value slider may say, as a multiple of what the rules imply. A log scale
# between them, because the derived table spans 1.25 to 74 on xiangqi and a linear
# +/-2 control is useless there -- what a person is saying is "the elephant is worth
# five times what the engine thinks", which is a statement they can make and check.
VALUE_MULTIPLIER_MIN = 0.1
VALUE_MULTIPLIER_MAX = 10.0


class DesignError(ValueError):
    """A game or an opponent that cannot be built as asked."""


# ------------------------------------------------------------------ starting points


def templates() -> list[dict[str, Any]]:
    """What a new game can start from, shortest first.

    A blank spec is not a starting point -- the first thing it does is fail validation
    for having no pieces -- and neither is chess, which is forty rules deep before you
    have changed anything. So: something minimal that already plays, something with a
    royal piece and a checkmate, and something won by arriving somewhere. Each is a
    complete, valid, playable spec, which is the property that matters: the editor's
    first draw is a board with pieces on it rather than a list of errors.

    **A template also passes the screen**, so a draft starts on a game a searching player
    can win and every rung on the page reads green before anything is changed.
    `TestBuildingAGameByHand::test_every_template_passes_the_screen` holds that.
    """
    return [
        {
            "key": "skirmish",
            "title": "a skirmish",
            "means": "One piece type on a small board, won by taking everything. "
                     "A capture you can make is a capture you must make, which is "
                     "what stops the two lines refusing each other.",
            "spec": _skirmish(),
        },
        {
            "key": "kingly",
            "title": "a game with a king",
            "means": "Two piece types and a royal one, won by checkmate. Add pieces "
                     "to it and you are writing a chess variant.",
            "spec": _kingly(),
        },
        {
            "key": "race",
            "title": "a race",
            "means": "Won by getting a piece to the far rank rather than by capture, "
                     "which is the quickest way out of chess's shadow.",
            "spec": _race(),
        },
    ]


def _skirmish() -> dict[str, Any]:
    """Two lines of soldiers, compulsory capture, won by taking everything.

    **`must_capture` is what makes this a game rather than a standoff.** Without it, a
    soldier that steps into diagonal range of another simply loses itself, so neither
    search ever opens the contact -- both shuffle sideways and all eight of the screen's
    games draw on the no-progress clock at exactly ply 40, with not one capture played
    (measured 2026-09-14, Q220). Compulsory capture takes the refusal away: once the
    lines touch, the take is forced, and the trades that follow are the game.
    """
    return {
        "schema_version": 2,
        "name": "skirmish",
        "description": "Two lines of soldiers on a six-by-six board. A soldier steps "
                       "forward or sideways and takes on the diagonal, a capture you can "
                       "make is one you must make, and taking every enemy piece wins.",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {
                "id": "S", "name": "soldier", "value": 1,
                "moves": [
                    {"kind": "leap", "vectors": [[0, 1], [1, 0]], "symmetry": "lateral",
                     "mode": "move"},
                    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "dihedral8",
                     "mode": "capture"},
                ],
            },
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "S", "squares": ["a1", "b1", "c1", "d1", "e1", "f1"]},
        ]},
        "rules": {"win": "annihilation", "stalemate": "draw", "max_plies": 160,
                  "repetition_draw": 3, "no_progress_draw": 40, "must_capture": True},
    }


def _kingly() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "small king game",
        "description": "Pawns, two rooks and a king a side on a six-by-six board, won by "
                       "checkmate. A pawn that reaches the far rank becomes a rook.",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {
                "id": "P", "name": "pawn", "value": 1,
                "moves": [
                    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral",
                     "mode": "capture"},
                ],
            },
            {
                "id": "R", "name": "rook", "value": 5,
                "moves": [{"kind": "slide", "vectors": [[0, 1]], "symmetry": "rotate4"}],
            },
            {
                "id": "K", "name": "king", "royal": True, "value": 0,
                "moves": [
                    {"kind": "leap", "vectors": [[0, 1], [1, 1]], "symmetry": "dihedral8"},
                ],
            },
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "R", "squares": ["a1", "f1"]},
            {"piece": "K", "squares": ["d1"]},
            {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2", "f2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 200,
                  "repetition_draw": 3, "no_progress_draw": 50,
                  "promotion": {"piece": "P", "rank": "last", "to": ["R"]}},
    }


def _race() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "name": "race",
        "description": "Runners that jump like knights or step one square forward, and "
                       "guards that slide up to two squares on the diagonal. The first runner to reach the far rank wins; "
                       "nothing is won by capturing.",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {
                "id": "R", "name": "runner", "value": 3,
                "moves": [
                    {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"},
                    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                ],
            },
            {
                "id": "G", "name": "guard", "value": 2,
                "moves": [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "dihedral8",
                           "range": 2}],
            },
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "R", "squares": ["b1", "e1"]},
            {"piece": "G", "squares": ["a1", "c1", "d1", "f1"]},
        ]},
        "rules": {
            "win": "none", "stalemate": "draw", "max_plies": 200,
            "repetition_draw": 3, "no_progress_draw": 60,
            # The whole point of this template: it is won by arriving, not by capturing.
            "reach": {"piece": "R", "squares": ["a6", "b6", "c6", "d6", "e6", "f6"],
                      "mirror": True, "result": "win", "safe": False},
        },
    }


# ------------------------------------------------------------------ pieces from the library
#
# Option D1 of `docs/plan-designer-ux.md`: every distinct piece the library holds, with
# its moves attached, so a piece that already plays can be lifted into a draft instead of
# retyped. Three of the four walks entered the same king's step two or three times over,
# each one starting from a new piece's default and being retyped (finding 2.6).
#
# **No piece is named here.** What a piece *is* comes out of the specs; the names, the
# letters and the counts are read off them as data. A palette entry carries only the
# self-contained half of a piece, because `captures`, `promoted_from` and `region` name
# pieces and squares of the game it came out of and mean nothing in a draft.

# What makes two games' pieces the same piece. Not the letter, the name or the worth --
# those are spelling, and the same piece is called three things across the library.
_PIECE_MECHANICS = ("moves", "royal", "facing")

# What travels into the draft when one is taken.
_PIECE_CARRIED = ("name", "value", "royal", "moves", "facing")


def piece_palette(corpus: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every distinct piece in the library, with its moves and the games that use it.

    Keyed by mechanics and by lattice, because a vector on a hex board points somewhere
    else: a piece is offered to a draft on the lattice it was written for and nowhere
    else. Each entry carries the sentences `spec/describe.py` says its moves in, so the
    tray reads as English and the page invents no vocabulary of its own.
    """
    found: dict[str, dict[str, Any]] = {}
    for spec in corpus:
        lattice = ((spec.get("board") or {}).get("lattice")) or "square"
        game = str(spec.get("name") or "")
        for piece in spec.get("pieces") or []:
            if not isinstance(piece, dict) or not piece.get("moves"):
                continue
            mechanics = {key: piece[key] for key in _PIECE_MECHANICS if key in piece}
            key = json.dumps([lattice, mechanics], sort_keys=True)
            entry = found.get(key)
            if entry is None:
                entry = found[key] = {
                    "key": f"p{len(found)}",
                    "lattice": lattice,
                    "piece": {key: piece[key] for key in _PIECE_CARRIED if key in piece},
                    "says": [
                        move_sentence(move, lattice == "hex") for move in piece["moves"]
                    ],
                    "ids": [],
                    "names": [],
                    "games": [],
                }
            entry["ids"].append(str(piece.get("id") or ""))
            entry["names"].append(str(piece.get("name") or piece.get("id") or ""))
            if game and game not in entry["games"]:
                entry["games"].append(game)
    return sorted(
        (_palette_entry(entry) for entry in found.values()),
        key=lambda entry: (-len(entry["games"]), entry["name"]),
    )


def _palette_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """One palette row, named and lettered by what the library calls it most often."""
    from collections import Counter

    name = Counter(entry.pop("names")).most_common(1)[0][0]
    letter = Counter(entry.pop("ids")).most_common(1)[0][0]
    piece = dict(entry["piece"], name=name)
    return {**entry, "piece": piece, "name": name, "id": letter,
            "games": sorted(entry["games"])}


# ------------------------------------------------------------------ settling the stamp


def settled(spec: Any) -> Any:
    """The draft as the editor treats it: stamped with the schema version it needs.

    **The one point at which a draft's `schema_version` is decided**, and every entry
    into this module goes through it -- `check`, `move_map` and `save` -- so what the
    board draws, what the verdict says and what is written to the archive cannot be three
    different games.

    The stamp is a claim about which features a spec uses, and until now the editor made
    a person type it. Twenty-two of the twenty-six rule keys are reachable only through
    the JSON box, most of them arrived after schema 2, and the three templates declare 1,
    1 and 2 -- so pasting a `strike` block into a template was answered with *rules uses
    strike, added in schema 6; declare "schema_version": 6 to use it*, which is a good
    sentence about a number nobody should have to know.

    `settle_schema_version` asks the *validator* for the lowest version the whole spec
    passes under, so this cannot go stale on a `SCHEMA_VERSION` bump, and it settles in
    both directions: a draft that declares more than it uses is stamped down, exactly as
    an add-and-remove pair of operators is. That matters because **`schema_version` is
    inside the mechanics hash** -- two people who built the same game from two templates
    would otherwise save two games.

    Copied rather than settled in place: the caller's draft is the browser's, and a
    function that quietly rewrote it would make "what did I send" unanswerable.
    """
    import copy

    if not isinstance(spec, dict):
        return spec
    return settle_schema_version(copy.deepcopy(spec))


# ------------------------------------------------------------------ checking a draft


def check(spec: Any, parent: dict[str, Any] | None = None) -> dict[str, Any]:
    """Whether this draft is a game, and everything the editor can show about it.

    Three answers, and they are deliberately separate. *Valid* is the schema's -- a spec
    that does not validate cannot be compiled, so nothing below it runs. *Playable* is
    the cheap gate's, which plays a few hundred random games and reports that nobody can
    move, or that every game is drawn. And the rest is what the draft would look like: the
    rules in English, the opening position, the pieces with their icons.

    Errors are returned rather than raised. The editor asks after every keystroke that
    changes something, and a draft in the middle of being written is invalid most of the
    time -- that is not an exceptional condition, it is the normal one.

    The draft is `settled` first, so a rule pasted into the JSON box is never refused for
    a version stamp it could have been given -- and the version that comes back is part
    of the answer, because it is inside the hash the same answer reports.

    Given the game the draft was forked from, the answer also says what changed from it,
    in sentences (`describe_changes`). The parent is settled too, so a stamp the draft was
    only given here is not reported as a change a person made.
    """
    if not isinstance(spec, dict):
        return {"valid": False, "errors": ["That draft could not be read."]}

    declared = spec.get("schema_version")
    spec = settled(spec)

    # `validate_spec` *returns* its problems rather than raising them, which is what makes
    # it usable here: a draft halfway through being written is invalid most of the time,
    # and one problem per line is what the editor lists.
    problems = _worth_saying(validate_spec(spec))
    if problems:
        # With the squares a board change left off it, which the page removes and names
        # after somebody reshapes the board (Q281c3); a valid draft has none.
        # And with the draft's own board, drawn from a stand-in, so a board control is
        # seen to change the board while the rules are still not a game (Q640).
        preview = board_preview(spec)
        sides = (preview or {}).get("sides")
        return {"valid": False, "errors": plain_problems(problems, spec, sides),
                "offBoard": off_board(spec),
                "preview": preview,
                "moveLines": move_lines(spec),
                "portals": _portals_answer(spec, problems, preview)}

    # A spec the schema accepts can still fail to compile -- the compiler enforces things
    # the validator does not, and it says so by raising. Its message is one paragraph,
    # so it is split back into lines: eight things wrong with a draft is a list.
    try:
        engine = Engine(spec)
        state = engine.initial_state()
        moves = engine.legal_moves(state)
    except (SpecError, ValueError) as error:
        preview = board_preview(spec)
        return {"valid": False, "errors": _errors_of(error, spec),
                "preview": preview, "moveLines": move_lines(spec),
                "portals": _portals_answer(spec, [], preview)}

    import time

    from chess_variant_sandbox.loop.gates import smoke_test

    # Timed, because the designer states what each rung costs on this machine.
    smoke_started = time.perf_counter()
    try:
        gate = smoke_test(spec)
        playable, reasons = gate.passed, list(gate.reasons)
    except Exception as error:  # noqa: BLE001 - a draft must not crash the editor
        playable, reasons = None, [f"These rules couldn't be tried out: {error}."]
    smoke_seconds = round(time.perf_counter() - smoke_started, 3)
    reasons = [_with_the_counter_named(line, engine, spec) for line in reasons]
    preview = _preview(engine, state, spec)

    return {
        "valid": True,
        "errors": [],
        "hash": spec_hash(spec),
        # The stamp this draft settled at, and the one it was sent with. Reported rather
        # than applied silently because it is *inside* the hash above: a person who can
        # see the hash move has to be able to see why.
        "schemaVersion": spec.get("schema_version"),
        "declaredVersion": declared,
        "settledVersion": declared != spec.get("schema_version"),
        "playable": playable,
        "playableReasons": reasons,
        "smokeSeconds": smoke_seconds,
        "openingMoves": len(moves),
        "ruleLines": describe_rules(spec),
        # How each piece moves, in the sentences its game's own page reads (Q541). The
        # rule lines above are every rule *beyond* movement, so a piece taught a slide
        # along an oblique line -- a nightrider -- validated and was read back nowhere.
        "pieceLines": [_piece_line(piece) for piece in describe_pieces(spec)],
        "setupLines": describe_setup(spec),
        "moveLines": move_lines(spec),
        "changes": _changes_from(parent, spec),
        # Enough for the editor to draw the opening position with the board component the
        # rest of the page already has.
        "preview": preview,
        # Schema 19 (P3): the pairs' cards in sentences, their warnings with the exits each
        # end has (N4), and whether *+ portal pair* can add another.
        "portals": _portals_answer(spec, [], preview, engine),
    }


def move_lines(spec: Any) -> list[list[str]]:
    """Each typed way of moving read back in words, piece by piece, row by row (F10).

    The designer types a step as *sideways then forward* numbers, and a newcomer could not
    tell what ``1,1`` did (`reference/first-visit-study.md` F10). This is the sentence
    `move_sentence` says for the same descriptor on a game's page, one per editor row, in
    the editor's order. Asked of drafts that are not a game yet too, because a half-typed
    row is exactly when a person needs to know what it says: a row no sentence can be made
    of yet (no direction, a number still being typed) reads as an empty string, and the
    page shows nothing under it rather than a guess.
    """
    if not isinstance(spec, dict):
        return []
    hexy = (spec.get("board") or {}).get("lattice") == "hex"
    lines: list[list[str]] = []
    for piece in spec.get("pieces") or []:
        moves = piece.get("moves") if isinstance(piece, dict) else None
        said: list[str] = []
        for move in moves if isinstance(moves, list) else []:
            try:
                ready = isinstance(move, dict) and bool(move.get("vectors"))
                said.append(move_sentence(move, hexy) if ready else "")
            except Exception:  # noqa: BLE001 - a half-written row reads as nothing yet
                said.append("")
        lines.append(said)
    return lines


def _piece_line(piece: Any) -> str:
    """One piece as the designer's *rules, in English* panel lists it: who, then how."""
    royal = ", royal" if piece.royal else ""
    said = "; ".join([*piece.moves, *piece.notes]) or "it has no move"
    return f"{piece.id} ({piece.name}{royal}): {said}."


def off_board(spec: Any) -> list[dict[str, Any]]:
    """The named squares a board change left off the board, where each was named (Q281c3).

    A wall, an army's square or a promotion square that is outside the frame, or on a
    corner a hexagon does not have, is refused by the validator; after somebody shrinks
    the board or makes a rhombus a hexagon that is a refusal about squares they never
    touched. The page removes these and says so, and asks here which they are, because
    which squares a board has is geometry and the page computes none.

    One entry per list, with ``path`` the keys down to it, ``squares`` the names to take
    out of it, and ``why`` the words for them. A square is off when its second player's
    image is, as `schema._check_named_squares` reads an army or a promotion square; a wall
    is judged alone, as a hexagon's corners are. Malformed names are left for the
    validator to name. Empty for anything that is not a draft.
    """
    from chess_variant_sandbox.spec.schema import (
        MAX_BOARD_DIM,
        absent_squares,
        parse_square,
        reflect_square,
    )

    if not isinstance(spec, dict) or not isinstance(spec.get("board"), dict):
        return []
    board = spec["board"]
    width, height = board.get("width"), board.get("height")
    if not all(isinstance(size, int) and 1 <= size <= MAX_BOARD_DIM for size in (width, height)):
        return []
    absent = absent_squares(board)
    setup: dict[str, Any] = spec["setup"] if isinstance(spec.get("setup"), dict) else {}
    rules: dict[str, Any] = spec["rules"] if isinstance(spec.get("rules"), dict) else {}
    orientation = setup.get("orientation", "mirror")

    def off(name: Any, reflection: str | None) -> bool:
        if not isinstance(name, str) or parse_square(name, MAX_BOARD_DIM, MAX_BOARD_DIM) is None:
            return False
        at = parse_square(name, width, height)
        if at is None:
            return True
        cells = [at] + ([reflect_square(*at, width, height, reflection)] if reflection else [])
        return any(rank * width + file in absent for file, rank in cells)

    lists: list[tuple[list[Any], Any, str | None]] = []
    placements = setup.get("placements")
    army = orientation if setup.get("mirror", True) else None
    if isinstance(placements, list):
        lists += [(["setup", "placements", index, "squares"], one.get("squares"), army)
                  for index, one in enumerate(placements) if isinstance(one, dict)]
    promotion = rules.get("promotion")
    if isinstance(promotion, dict):
        lists.append((["rules", "promotion", "squares"], promotion.get("squares"), orientation))
    elif isinstance(promotion, list):
        lists += [(["rules", "promotion", index, "squares"], one.get("squares"), orientation)
                  for index, one in enumerate(promotion) if isinstance(one, dict)]
    obstacles = rules.get("obstacles")
    if isinstance(obstacles, dict):
        lists.append((["rules", "obstacles", "squares"], obstacles.get("squares"), None))

    why = "off the hexagon" if absent else "off the board"
    found = []
    for path, names, reflection in lists:
        if isinstance(names, list):
            gone = [name for name in names if off(name, reflection)]
            if gone:
                found.append({"path": path, "squares": gone, "why": why})
    return found


# ------------------------------------------------------------------ the draft's own board


# The pieces of a stand-in: every one keeps its letter, name and worth and steps one cell
# forward, which every lattice and every shape takes.
_STAND_IN_MOVE = {"kind": "leap", "vectors": [[0, 1]]}


def board_preview(spec: Any) -> dict[str, Any] | None:
    """The draft's own board and army, drawn while its rules are not a game yet (Q640).

    A board control changes the board at once: the cells, the hexagon's corners, the walls
    and where the army stands. Before this, an invalid draft drew the last board a check
    had made, so making the cells hexagons left the squares on screen, and with no such
    board (a reload, a phone restoring the tab) the board went away altogether.

    The rules that keep the draft from being a game are rarely about the board, so the
    board is drawn from a *stand-in*: the draft with every piece's moves replaced by one
    step forward, and on the hex lattice the setup turned as the hex lattice turns it.
    The army is the part of it that stands on the board (`_standing_army`). Asked with the
    draft's rules first, then with only its walls and seams, then with none; the first the
    validator and the compiler take is drawn, so the board is the compiler's, as every
    other board on the page is. None when no stand-in is a game either (a width of zero,
    a shape the lattice does not have, nobody standing on the board).
    """
    import copy

    if not isinstance(spec, dict) or not isinstance(spec.get("board"), dict):
        return None
    base = copy.deepcopy(spec)
    pieces = base.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        return None
    base["pieces"] = [
        {key: piece[key] for key in ("id", "name", "value", "royal") if key in piece}
        | {"moves": [dict(_STAND_IN_MOVE)]}
        for piece in pieces if isinstance(piece, dict)
    ]
    setup = dict(base["setup"]) if isinstance(base.get("setup"), dict) else {}
    # Where each piece stands and whose it is; a facing belongs to moves it no longer has.
    if isinstance(setup.get("placements"), list):
        setup["placements"] = [
            {key: row[key] for key in ("piece", "squares", "player") if key in row}
            for row in setup["placements"] if isinstance(row, dict)
        ]
    hexed = base["board"].get("lattice") == "hex"
    if hexed:
        setup = {**setup, "orientation": "rotate"}
    rules: dict[str, Any] = base["rules"] if isinstance(base.get("rules"), dict) else {}
    royal = any(piece.get("royal") for piece in base["pieces"])
    plain: dict[str, Any] = {"win": "royal_capture" if royal else "annihilation",
                             "max_plies": 100}
    walls_and_seams = {key: rules[key] for key in ("obstacles", "wrap") if key in rules}
    tries: list[dict[str, Any]] = [
        {**base, "setup": setup, "rules": rules},
        {**base, "setup": setup, "rules": {**plain, **walls_and_seams}},
        {**base, "setup": setup, "rules": plain},
    ]
    tries = [_standing_army(stand_in) for stand_in in tries]
    # Nobody left standing (a hexagon grown round an army on its corners): the board alone,
    # drawn from one piece stood on its first cell and then taken off the drawing again.
    nobody = not tries[-1]["setup"].get("placements")
    if nobody:
        tries = [{**stand_in, "setup": {**stand_in["setup"], "mirror": True,
                                        "placements": _one_placement(stand_in)}}
                 for stand_in in tries[1:]]
    for stand_in in tries:
        stand_in = settled(stand_in)
        if validate_spec(stand_in):
            continue
        try:
            engine = Engine(stand_in)
            state = engine.initial_state()
        except (SpecError, ValueError):
            continue
        preview = {**_preview(engine, state, stand_in), "standIn": True}
        # The draft's own wild pieces' walks, on the stand-in's board (Q852): drawn while
        # the rules are not a game yet too, so a refusal about a walk can be seen, though
        # the stand-in that compiled may be one that dropped them.
        preview.pop("walks", None)
        if walks := walk_payload(spec, engine.spec):
            preview["walks"] = walks
        if nobody:
            preview["board"] = [0] * len(preview["board"])
        return preview
    return None


def _one_placement(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The first piece on the first cell of the board that is a cell and no wall: the one
    placement a stand-in with nobody standing needs to be a spec at all."""
    from chess_variant_sandbox.spec.schema import absent_squares

    board = spec["board"]
    width, height = board.get("width"), board.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 2:
        return []
    obstacles = (spec.get("rules") or {}).get("obstacles")
    walls = set(obstacles.get("squares") or []) if isinstance(obstacles, dict) else set()
    absent = absent_squares(board)
    for square in range(width):
        name = _square_name(square, width)
        if square not in absent and name not in walls:
            return [{"piece": spec["pieces"][0]["id"], "squares": [name]}]
    return []


def _standing_army(spec: dict[str, Any]) -> dict[str, Any]:
    """A stand-in's army without what cannot stand: a square off the board or on a
    hexagon's absent corner (`off_board`, as the page drops them after a reshape), or a
    name that is no square at all. A row left empty goes; a stand-in with no row left is
    refused by the validator, since a board with nobody on it is no spec."""
    from chess_variant_sandbox.spec.schema import MAX_BOARD_DIM, parse_square

    setup = spec.get("setup")
    if not isinstance(setup, dict) or not isinstance(setup.get("placements"), list):
        return spec
    gone: dict[int, set[Any]] = {}
    for entry in off_board(spec):
        if entry["path"][:2] == ["setup", "placements"]:
            gone.setdefault(entry["path"][2], set()).update(entry["squares"])
    rows = []
    for index, row in enumerate(setup["placements"]):
        squares = row.get("squares") if isinstance(row, dict) else None
        if not isinstance(squares, list):
            rows.append(row)
            continue
        kept = [name for name in squares if name not in gone.get(index, set())
                and isinstance(name, str)
                and parse_square(name, MAX_BOARD_DIM, MAX_BOARD_DIM) is not None]
        if kept:
            rows.append({**row, "squares": kept})
    return {**spec, "setup": {**setup, "placements": rows}}


def opening_preview(spec: dict[str, Any]) -> dict[str, Any]:
    """A library game's opening position, shaped as `check`'s `preview` (the front page's
    first step draws it with the same still board, Q564)."""
    from chess_variant_sandbox.web.library import engine_for

    engine = engine_for(spec)
    return _preview(engine, engine.initial_state(), spec)


# ------------------------------------------------------------------ portals in the designer
#
# P3 of `docs/archive/plan-portals.md` (Q772c): the pair's card in sentences (D4), the refusals in
# the designer's words (D7), the warnings when a setting makes a portal useless (D7, N4),
# and whether another pair can go in at all (D2). The sentences say what the rule *is*,
# from the spec's own fields through a table of phrasings, never what a position allows;
# the one count read off a board -- how many squares an end can let a piece out onto -- is
# the compiled exit table's (`PortalRule.exits`), so nothing here works out a square.
#
# An end is named by its square, never by which end of the pair it is: both ends of a pair
# are drawn alike, in the pair's colour (Q838d, `docs/archive/plan-portal-look.md` §5). The file's
# keys `orange` and `blue` stay; they are data, not words. Where an end has no square to
# name (it is off the board), it is the pair's *first* or *second* end, the order the
# designer put them down in.
_END_ORDER = {"orange": "first", "blue": "second"}


def _end_at(pair: dict[str, Any], end: str) -> str:
    """An end as a sentence names it: *the end on b2*."""
    square = pair.get(end)
    return f"the end on {square}" if isinstance(square, str) and square else \
        f"the {_END_ORDER[end]} end"

# The move a portal piece is given the first time somebody may move an end: one step in
# any direction, onto an empty square (D4). The same both ways up, as a portal's must be.
PORTAL_STEP: dict[str, Any] = {
    "kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "move",
}

# What a rule beside which portals are refused is called, where its card's own title would
# not read in the sentence (D7). Every other rule is named by `RULE_WORDS`.
_PORTAL_BESIDE: dict[str, str] = {}


def _whose(side: str) -> str:
    """A side's possessive, as a sentence opens with it: *White's*, *the first player's*."""
    return f"{side.capitalize()}'s" if side in ("white", "black") else f"the {side} player's"


def _sides_of(sides: Any) -> tuple[str, str]:
    if isinstance(sides, (list, tuple)) and len(sides) == 2:
        return (str(sides[0]), str(sides[1]))
    return ("white", "black")


def _movers_words(movers: Any, sides: tuple[str, str]) -> str:
    """Who may move an end, as the subject of a sentence."""
    players = sorted(movers)
    if not players:
        return "nobody"
    if len(players) > 1:
        return "either player"
    side = sides[players[0]]
    return side.capitalize() if side in ("white", "black") else f"the {side} player"


def _pair_view(group: dict[str, Any]) -> dict[str, Any]:
    """A group written as a pair, in the shape this module's sentences read: the two ends'
    squares under ``orange`` and ``blue``, and who moves each by the same names."""
    ends = group["ends"]
    return {
        "piece": group["piece"],
        "orange": ends[0]["at"],
        "blue": ends[1]["at"],
        "passage": group["passage"],
        "moved_by": {end: ends[side]["moved_by"] for side, end in enumerate(PORTAL_ENDS)},
        "no_straight_back": group["no_straight_back"],
    }


def _piece_plural(name: str) -> str:
    """A piece's name, many of it, as the page's `portalPlural` says it."""
    if re.search(r"(s|x|z|ch|sh)$", name, re.I):
        return name + "es"
    if re.search(r"[^aeiou]y$", name, re.I):
        return name[:-1] + "ies"
    return name + "s"


def _list_words(names: list[str]) -> str:
    """*a*, *a and b*, *a, b and c*."""
    return names[0] if len(names) < 2 else f"{', '.join(names[:-1])} and {names[-1]}"


def _filter_sentence(spec: Any, carries: Any, at: str) -> str:
    """Which pieces may go in at an end, by their names (schema 21's ``carries``, §5.2)."""
    names = [_piece_plural(_piece_called(spec, pid)) for pid in sorted(carries)]
    if not names:
        return f"No piece may go in at {at} yet."
    return f"Only {_list_words(names)} may go in at {at}."


def _group_called(group: dict[str, Any]) -> str:
    """A group as a sentence names it: *pair 1* for two ends, *group 2* for more."""
    return f"{'pair' if len(group['ends']) == 2 else 'group'} {group['number']}"


def portal_cards(spec: Any, sides: Any = None, engine: Any = None) -> list[dict[str, Any]]:
    """One card per group, in the designer's sentences (D4) with its warnings (D7, N4).

    Read off ``rules.portals`` as the validator and the compiler read it (`portal_groups`),
    so a group means one thing to all three. ``engine``, where the draft compiles, gives the
    exits each end has on the board as set up (N4); without one the cards carry no counts
    and so no warning about them. Empty for a draft without portals. A pair's card is the
    one it has always had, with its filters said; a group of three or more ends (schema 21,
    `docs/archive/plan-portals-2.md` §5.4, Q843b) says where each end leads, whose pieces go in by
    it, which pieces, and who moves it (`_map_card`). Each card carries its group's number
    (``pair``), which is how the page finds it.
    """
    if not isinstance(spec, dict):
        return []
    rules = spec.get("rules")
    groups = portal_groups(rules)
    if not groups:
        return []
    named = _sides_of(sides)
    exits = _portal_exit_counts(engine)
    piece_moves = _portal_piece_moves(spec)
    cards: list[dict[str, Any]] = []
    first = 0
    for group in groups:
        start, first = first, first + len(group["ends"])
        if group["spelling"] != "pair":
            cards.append(_fold_card(group, named,
                                    _map_card(spec, group, named, exits[start: first], piece_moves)))
            continue
        index = group["number"] - 1
        pair = _pair_view(group)
        passage = [pair["passage"][0], pair["passage"][1]]
        same = passage[0] == passage[1]
        ways = _passage_sentences(passage, named, pair)
        movers = {end: sorted(pair["moved_by"][end]) for end in PORTAL_ENDS}
        moves = _moved_by_sentence(movers, named, piece_moves.get(pair["piece"]), pair)
        if pair["no_straight_back"] and any(movers.values()):
            # Schema 20 (Q807): the bar, said where who may move the pair is said.
            moves += (" A portal that has just been moved can't be moved straight back where "
                      "it came from on the very next move.")
        counts = exits[start: start + 2] if len(exits) >= start + 2 else []
        cards.append({
            "pair": index + 1,
            "piece": pair["piece"],
            "orange": pair["orange"],
            "blue": pair["blue"],
            "passage": passage,
            "same": same,
            "movedBy": movers,
            "noStraightBack": pair["no_straight_back"],
            "ways": ways["ways"],
            "solid": ways["solid"],
            "moves": moves,
            "exits": dict(zip(PORTAL_ENDS, counts, strict=True)) if counts else None,
            "warnings": _portal_warnings(passage, counts, pair),
            "spelling": "pair",
            "filters": [_filter_sentence(spec, end["carries"], _end_at(pair, key))
                        for key, end in zip(PORTAL_ENDS, group["ends"], strict=True)
                        if end["carries"] is not None],
        })
        cards[-1] = _fold_card(group, named, cards[-1])
    return cards


def _fold_card(group: dict[str, Any], sides: tuple[str, str], card: dict[str, Any]) -> dict[str, Any]:
    """A group's card with its kind (schema 22, Q843d, `docs/archive/plan-portals-2.md` §5.4): a
    momentum group's card as it was, with ``kind`` said; a fold's ways in a fold's words. A
    fold is not a door a piece is thrown through: a piece that ends its move on it may stay,
    or come out on the end it leads to (schema 24, `docs/archive/plan-portals-2.md` §5.3R), and one
    a piece may not use is plain floor to it, never solid. Nothing comes out *beside* a
    fold, so the momentum card's exit counts and their warnings do not apply. Ends by
    square, as every card names them."""
    if group["kind"] != "fold":
        return {**card, "kind": "momentum"}
    ends = group["ends"]
    at = [f"the end on {end['at']}" if isinstance(end["at"], str) and end["at"]
          else f"end {place + 1}" for place, end in enumerate(ends)]
    both = "White's and Black's" if sides == ("white", "black") else "both players'"
    ways: list[str] = []
    for place, end in enumerate(ends):
        if end["to"] is not None and end["enter"]:
            who = both if len(end["enter"]) == 2 else _whose(sides[min(end["enter"])])
            ways.append(f"{who} pieces that stop on {at[place]} may stay there, or come out "
                        f"on {at[end['to']]}.")
            continue
        sources = [at[other] for other, one in enumerate(ends) if one["to"] == place]
        if sources:
            ways.append(f"{at[place].capitalize()} leads nowhere: nothing goes through it, "
                        "and pieces come out of it from "
                        f"{_list_words(sources).replace('the end on ', '')}.")
    moves = str(card.get("moves") or "").replace(
        ", always onto an empty square", ", onto an empty square or under a piece")
    return {**card, "kind": "fold", "ways": ways, "solid": [], "moves": moves,
            "warnings": [], "exits": None}


def _map_card(
    spec: Any, group: dict[str, Any], sides: tuple[str, str], counts: list[int],
    piece_moves: dict[str, list[str]],
) -> dict[str, Any]:
    """A group of three or more ends (schema 21's ``ends``), in the card's sentences: where
    each end leads and whose pieces go in by it, which ends are only ways out, who may move
    each, which pieces each lets in, and what makes an end useless (N4). Ends by square."""
    ends = group["ends"]
    at = [f"the end on {end['at']}" if isinstance(end["at"], str) and end["at"]
          else f"end {place + 1}" for place, end in enumerate(ends)]
    both = "White's and Black's" if sides == ("white", "black") else "both players'"
    ways: list[str] = []
    solid: list[str] = []
    for place, end in enumerate(ends):
        if end["to"] is not None:
            who = both if len(end["enter"]) == 2 else _whose(sides[min(end["enter"])])
            ways.append(f"{who} pieces can go in at {at[place]} and come out by {at[end['to']]}.")
            for player in (0, 1):
                if player not in end["enter"]:
                    solid.append(f"{at[place].capitalize()} is solid to {_whose(sides[player])} "
                                 "pieces: they can't stand on it, take it, or slide through it.")
            continue
        sources = [at[other] for other, one in enumerate(ends) if one["to"] == place]
        if sources:
            ways.append(f"Nothing goes in at {at[place]}: pieces come out by it from "
                        f"{_list_words(sources).replace('the end on ', '')}.")
            solid.append(f"{at[place].capitalize()} is solid to every piece: nothing can stand "
                         "on it, take it, or slide through it.")
    movers = [sorted(end["moved_by"]) for end in ends]
    words = [_movers_words(players, sides) for players in movers]
    if len(set(words)) == 1:
        moves = ("Nobody moves these portals." if words[0] == "nobody"
                 else f"{words[0].capitalize()} may spend a turn moving any of them.")
    else:
        moves = " ".join(f"Nobody moves {at[place]}." if who == "nobody"
                         else f"{who.capitalize()} may spend a turn moving {at[place]}."
                         for place, who in enumerate(words))
    if any(movers):
        ways_moved = [way.removesuffix(", not to take") for way in piece_moves.get(group["piece"]) or []]
        how = ("; ".join(ways_moved) + ", always onto an empty square" if ways_moved
               else "not at all yet: the portal piece has no move")
        moves += (f" A portal moves {how}. To change how portals move, open the portal on the "
                  "piece list.")
        if group["no_straight_back"]:
            moves += (" A portal that has just been moved can't be moved straight back where "
                      "it came from on the very next move.")
    warnings = []
    for place in range(len(ends)):
        led = any(one["to"] == place for one in ends)
        if counts and len(counts) == len(ends) and not counts[place] and led:
            warnings.append(f"Nothing can come out of {at[place]}: every square beside it is off "
                            "the board or a wall.")
    return {
        "pair": group["number"],
        "piece": group["piece"],
        "spelling": "ends",
        "shape": portal_map_shape(group),
        "ends": [{"at": end["at"], "to": end["to"], "enter": sorted(end["enter"]),
                  "movedBy": movers[place]} for place, end in enumerate(ends)],
        "noStraightBack": group["no_straight_back"],
        "ways": ways,
        "solid": solid,
        "moves": moves,
        "exits": counts or None,
        "warnings": warnings,
        "filters": [_filter_sentence(spec, end["carries"], at[place])
                    for place, end in enumerate(ends) if end["carries"] is not None],
    }


def _passage_sentences(
    passage: list[str], sides: tuple[str, str], pair: dict[str, Any]
) -> dict[str, list[str]]:
    """Which way pieces go through, one sentence per player (one when both are the same),
    and what *solid* means wherever an end is solid to somebody (D4). Ends by square."""
    same = passage[0] == passage[1]
    whose = (["White's and Black's" if sides == ("white", "black")
              else "both players'"] if same else [_whose(side) for side in sides])
    words = passage[:1] if same else passage
    ways: list[str] = []
    solid: list[str] = []
    for who, word in zip(whose, words, strict=True):
        if word == "both_ways":
            ways.append(f"{who} pieces can go in at either end and come out of the other.")
        elif word in ("orange_to_blue", "blue_to_orange"):
            into, out = ("orange", "blue") if word == "orange_to_blue" else ("blue", "orange")
            ways.append(f"{who} pieces can go in at {_end_at(pair, into)} and come out by "
                        f"{_end_at(pair, out)}, not back.")
            solid.append(f"{_end_at(pair, out).capitalize()} is solid to {who} pieces: they "
                         "can't stand on it, take it, or slide through it.")
        else:
            ways.append(f"{who} pieces can't use these portals: to them both ends are solid "
                        "blocks, like a wall.")
            solid.append(f"To {who} pieces both ends are solid: they can't stand on them, take "
                         "them, or slide through them.")
    return {"ways": ways, "solid": solid}


def _moved_by_sentence(
    movers: dict[str, list[int]], sides: tuple[str, str], moves: list[str] | None,
    pair: dict[str, Any],
) -> str:
    """Who may move each end, as one sentence for both, and how a portal moves where
    anybody may (D4). Ends by square."""
    first, second = (_movers_words(movers[end], sides) for end in PORTAL_ENDS)
    if first == second:
        said = ("Nobody moves these portals." if first == "nobody"
                else f"{first.capitalize()} may spend a turn moving either end.")
    else:
        parts = []
        for end, who in zip(PORTAL_ENDS, (first, second), strict=True):
            parts.append(f"Nobody moves {_end_at(pair, end)}." if who == "nobody"
                         else f"{who.capitalize()} may spend a turn moving {_end_at(pair, end)}.")
        said = " ".join(parts)
    if first == "nobody" and second == "nobody":
        return said
    # A portal's every move is *move only* (the validator's), so the sentence says where it
    # may go -- an empty square -- once, rather than *not to take* after each way.
    ways = [way.removesuffix(", not to take") for way in moves or []]
    how = ("; ".join(ways) + ", always onto an empty square" if ways
           else "not at all yet: the portal piece has no move")
    return (f"{said} A portal moves {how}. To change how portals move, open the portal on "
            "the piece list.")


def _portal_piece_moves(spec: dict[str, Any]) -> dict[str, list[str]]:
    """Each portal piece's ways of moving, in the sentences a game's page says them in."""
    hexy = (spec.get("board") or {}).get("lattice") == "hex"
    out: dict[str, list[str]] = {}
    for piece in spec.get("pieces") or []:
        if not isinstance(piece, dict) or not isinstance(piece.get("id"), str):
            continue
        said = []
        for move in piece.get("moves") or []:
            try:
                if isinstance(move, dict) and move.get("vectors"):
                    said.append(move_sentence(move, hexy))
            except Exception:  # noqa: BLE001 - a half-written row reads as nothing yet
                continue
        out[piece["id"]] = said
    return out


def _portal_exit_counts(engine: Any) -> list[int]:
    """How many squares each end can let a piece out onto, in end order (N4).

    The compiled exit table's answer, for each end where the setup puts it: the squares one
    king's step away under either player's wrap permission, off the board and walls being
    no square. Python counts a table the compiler built; it works out no square (rules 3
    and 4). Empty without a compiled draft.
    """
    portals = getattr(getattr(engine, "spec", None), "portals", None)
    if portals is None:
        return []
    counts = []
    for square in portals.squares:
        found = {out for player in (0, 1) for out in portals.exits[player][square] if out >= 0}
        counts.append(len(found))
    return counts


def _portal_warnings(passage: list[str], counts: list[int], pair: dict[str, Any]) -> list[str]:
    """What the card says when a setting is legal but makes a portal useless (D7)."""
    if passage[0] == "closed" and passage[1] == "closed":
        return ["Nobody can go through these portals, so they are just two solid blocks. If "
                "that is what you want, a wall does the same and is simpler."]
    warnings = []
    for side, end in enumerate(PORTAL_ENDS):
        if not counts or counts[side]:
            continue
        # Somebody comes out of this end only where somebody may go in at the other.
        other = 1 - side
        entered = any(_opens(word)[other] for word in passage)
        if entered:
            warnings.append(f"Nothing can come out of {_end_at(pair, end)}: every square "
                            "beside it is off the board or a wall.")
    return warnings


def _opens(word: str) -> tuple[bool, bool]:
    from chess_variant_sandbox.spec.schema import portal_open_ends

    return portal_open_ends(word)


def portal_tool(spec: Any) -> dict[str, Any]:
    """Whether *+ portals* can add another group to this draft, and if not, why (D2).

    Asked of the validator: a trial draft with one more pair, whose squares do not matter,
    since only the refusals that are about the game around the portals are read. The
    reason is the designer's sentence for that refusal (`portal_problem`), for the button's
    tooltip.
    """
    import copy

    if not isinstance(spec, dict):
        return {"open": False, "why": ""}
    rules: dict[str, Any] = spec["rules"] if isinstance(spec.get("rules"), dict) else {}
    held: list[Any] = rules["portals"] if isinstance(rules.get("portals"), list) else []
    if len(held) >= MAX_PORTAL_GROUPS:
        return {"open": False, "why": "A game can have at most four groups of portals."}
    if sum(len(group["ends"]) for group in portal_groups(rules)) + 2 > MAX_PORTAL_ENDS:
        return {"open": False, "why": "A game can have at most eight portals."}
    trial = copy.deepcopy(spec)
    trial_rules = trial.setdefault("rules", {})
    if not isinstance(trial_rules, dict):
        return {"open": False, "why": ""}
    pieces: list[Any] = trial["pieces"] if isinstance(trial.get("pieces"), list) else []
    taken = {piece.get("id") for piece in pieces if isinstance(piece, dict)}
    taken |= set(portal_end_ids(portal_groups(
        {"portals": [*held, {"piece": "", "orange": "a1", "blue": "b1"}]}
    )))
    used = {pair.get("piece") for pair in held if isinstance(pair, dict)}
    piece = next(iter(sorted(pid for pid in used if isinstance(pid, str))), None)
    if piece is None:
        piece = next(letter for letter in "OPQVWXYZ" + "".join(map(str, range(10)))
                     if letter not in taken)
        pieces.append({"id": piece, "name": "portal", "value": 0, "moves": []})
        trial["pieces"] = pieces
    trial_rules["portals"] = [*held, {"piece": piece, "orange": "a1", "blue": "b1"}]
    for problem in validate_spec(settled(trial)):
        said = portal_problem(problem, trial)
        if said and said.get("beside"):
            return {"open": False, "why": said["says"]}
    return {"open": True, "why": ""}


# The validator's portal refusals, each with the designer's sentence (D7). A pattern's
# groups are read by the function beside it; each answers {says, pair, end, square, beside}.
_P_PAIR = r"rules\.portals\[(\d+)\]"


def portal_problem(problem: str, spec: Any = None, sides: Any = None) -> dict[str, Any] | None:
    """One validator problem about portals in the designer's words (D7), or ``None``.

    ``pair`` (from 1) and ``end`` name the end at fault where there is one, so the card
    can show the sentence and *show me* can pick the end; ``beside`` marks a refusal about
    the game around the portals, which is what disables *+ portal pair*. Never drops or
    adds a problem: the caller says every other one through `spec.plain`.
    """
    named = _sides_of(sides)
    text = problem.strip()
    groups = {group["index"]: group for group in portal_groups(
        spec.get("rules") if isinstance(spec, dict) else None)}

    def called(index: int) -> str:
        """*Pair 1* or *Group 2*, as the group's size says (schema 21)."""
        group = groups.get(index)
        return (_group_called(group) if group else f"pair {index + 1}").capitalize()

    def square_of(index: int, place: int) -> str:
        group = groups.get(index)
        ends = group["ends"] if group else []
        at = ends[place]["at"] if place < len(ends) else None
        return at if isinstance(at, str) and at else "?"

    end_at = re.match(_P_PAIR + r"\.(orange|blue|ends\[(\d+)\]) (\S+) (.*)$", text)
    if end_at:
        index, end, square, rest = int(end_at.group(1)), end_at.group(2), end_at.group(4), \
            end_at.group(5)
        pair, who = index + 1, called(index)
        place = {"pair": pair, "end": end, "square": square,
                 "place": int(end_at.group(3)) if end_at.group(3) else PORTAL_ENDS.index(end)}
        # An end is named by its square (Q838d): both ends of a pair look alike.
        if rest == "is a wall or not on the board's shape":
            return {**place, "says": f"{who} has an end on a wall ({square}). Move it "
                    "to an empty square."}
        if rest == "is a square setup.placements puts a piece on":
            return {**place, "says": f"{who} has an end on {square}, where a piece "
                    "starts. A portal needs an empty square: move it, or take the piece off "
                    "first."}
        other = re.match(r"is also " + _P_PAIR + r"\.(orange|blue|ends\[\d+\]); two portals on "
                         r"one square$", rest)
        if other:
            return {**place, "says": f"{who} has an end on {square}, where "
                    f"{called(int(other.group(1)))[0].lower() + called(int(other.group(1)))[1:]} "
                    "has one too. Two portals can't share a square."}
    # Schema 21's refusals of a group of three or more ends (D7's new rows, Q843b).
    map_end = re.match(_P_PAIR + r"\.ends\[(\d+)\](\.\w+)? (.*)$", text)
    if map_end:
        index, end_place, rest = int(map_end.group(1)), int(map_end.group(2)), map_end.group(4)
        square = square_of(index, end_place)
        place = {"pair": index + 1, "end": f"ends[{end_place}]", "place": end_place,
                 "square": square}
        who = called(index)
        if rest.startswith("leads nowhere and no end leads to it"):
            return {**place, "says": f"{who}'s end on {square} leads nowhere and no end leads "
                    "to it, so no piece can use it. Lead it to another end, or lead another end "
                    "to it."}
        if rest.startswith("is the end itself"):
            return {**place, "says": f"{who}'s end on {square} leads to itself. Choose another "
                    "end for it to lead to."}
        if rest.startswith("must be the index of another end"):
            return {**place, "says": f"{who}'s end on {square} leads to an end that isn't in "
                    "the group. Choose one of its own ends."}
        also = re.match(r"(\S+) is also .*ends\[\d+\]; a group's ends need squares", rest)
        if also:
            return {**place, "square": also.group(1), "says": f"{who} has two ends on "
                    f"{also.group(1)}. Each end goes on a square of its own."}
        off_map = re.match(r"must be a square on the board, got (.*)$", rest)
        if off_map:
            return {**place, "says": f"{who} has an end that "
                    f"{_off_board_words(off_map.group(1))}. Put it on an empty square."}
    filtered = re.match(_P_PAIR + r"\.(?:carries\.(orange|blue)|ends\[(\d+)\]\.carries) (.*)$",
                        text)
    if filtered:
        index, rest = int(filtered.group(1)), filtered.group(4)
        end_place = (PORTAL_ENDS.index(filtered.group(2)) if filtered.group(2)
                     else int(filtered.group(3)))
        square = square_of(index, end_place)
        place = {"pair": index + 1, "place": end_place, "square": square,
                 "end": filtered.group(2) or f"ends[{end_place}]"}
        if rest.startswith("names every piece that can move"):
            return {**place, "says": f"The end on {square} lets every piece in, so its filter "
                    "keeps nothing out. Untick the pieces it keeps out, or choose all."}
        if rest.startswith("is empty"):
            return {**place, "says": f"The end on {square} has no piece ticked, so nothing "
                    "could go in by it. Tick the pieces it lets in, or choose all."}
        if "nobody may go in by this end" in rest or "the end leads nowhere" in rest:
            return {**place, "says": f"Nobody goes in by the end on {square}, so it has no "
                    "pieces to choose. Choose all, or let somebody go in there."}
    off = re.match(_P_PAIR + r"\.(orange|blue) must be a square on the board, got (.*)$", text)
    if off:
        pair, end = int(off.group(1)) + 1, off.group(2)
        return {"pair": pair, "end": end, "says": f"Pair {pair}'s {_END_ORDER[end]} end "
                f"{_off_board_words(off.group(3))}. Put it on an empty square."}
    both = re.match(_P_PAIR + r" puts both ends on (\S+); a pair needs two squares$", text)
    if both:
        pair = int(both.group(1)) + 1
        return {"pair": pair, "end": "blue", "square": both.group(2), "says": f"Pair {pair} "
                f"has both ends on {both.group(2)}. The second end goes on a different square."}
    barred = re.match(_P_PAIR + r"\.no_straight_back is true but nobody may move", text)
    if barred:
        pair = int(barred.group(1)) + 1
        return {"pair": pair, "says": f"Pair {pair} says a portal can't be moved straight "
                "back, but nobody may move its portals. Let somebody move them, or untick "
                "'not straight back'."}
    never_met = re.match(
        _P_PAIR + r"\.no_straight_back is true but it could never bar a move: (.*)$", text
    )
    if never_met:
        pair = int(never_met.group(1)) + 1
        if never_met.group(2).startswith("it binds only"):
            return {"pair": pair, "says": f"Pair {pair} says a portal can't be moved straight "
                    "back, but each end is moved by one player only, so the next move is always "
                    "the other player's and the rule never applies. Let both players move an "
                    "end, or untick 'not straight back'."}
        return {"pair": pair, "says": f"Pair {pair} says a portal can't be moved straight back, "
                "but the portal piece has no move back, so it could never go straight back "
                "anyway. Give the portal piece a move in the opposite direction, or untick "
                "'not straight back'."}
    if re.match(r"rules\.portals holds \d+ groups; at most", text):
        return {"says": "A game can have at most four groups of portals.", "beside": True}
    if re.match(r"rules\.portals has \d+ ends; at most", text):
        return {"says": "A game can have at most eight portals.", "beside": True}
    if re.match(rf"at most {MAX_PIECE_TYPES} piece types, and rules\.portals adds one", text):
        return {"says": "This game has too many kinds of piece for more portals: each end "
                "counts as one.", "beside": True}
    if text.startswith("rules.portals is refused on a hex lattice"):
        return {"says": "Portals can't be used on a hexagon board yet. Take off one of them: "
                "the portals, or the hexagons.", "beside": True}
    beside = re.match(r"rules\.portals is refused beside rules\.(\w+):", text)
    if beside:
        key = beside.group(1)
        title = _PORTAL_BESIDE.get(key) or RULE_WORDS.get(key, (key.replace("_", " "), ""))[0]
        return {"says": f"Portals can't be used together with the rule '{title}' yet. Take "
                "off one of them.", "beside": True}
    if text.startswith("rules.portals is refused beside a 'sweep' move"):
        return {"says": "Portals can't be used together with a piece that sweeps yet. Take off "
                "one of them.", "beside": True}
    region = re.match(r"piece '([^']*)' has a region, which rules\.portals refuses", text)
    if region:
        return {"says": f"Portals can't be used together with a piece kept to some squares "
                f"({_piece_called(spec, region.group(1))}'s 'only on these squares') yet. Take "
                "off one of them.", "beside": True}
    return _portal_piece_problem(text, spec, named)


# Where a portal piece was named that it may not be, as the sentence names the place.
_PORTAL_ELSEWHERE: tuple[tuple[str, str], ...] = (
    ("be placed by setup.placements", "where they start"),
    ("promote or be promoted into", "the rule 'promotion'"),
    ("be named by rules.drops", f"the rule '{RULE_WORDS['drops'][0]}'"),
    ("be rules.reach's piece", f"the rule '{RULE_WORDS['reach'][0]}'"),
    ("be counted by rules.extinction", f"the rule '{RULE_WORDS['extinction'][0]}'"),
    ("castle", "the rule 'castling'"),
)

_ONLY_FOR_PORTALS = ("The portal piece is only for portals: it can't start on the board, be "
                     "promoted to, or be taken. Remove it from {where}.")


def _piece_called(spec: Any, pid: str) -> str:
    pieces = spec.get("pieces") if isinstance(spec, dict) else None
    for piece in pieces if isinstance(pieces, list) else []:
        if isinstance(piece, dict) and piece.get("id") == pid:
            return str(piece.get("name") or pid)
    return pid


def _portal_piece_problem(
    text: str, spec: Any, sides: tuple[str, str]
) -> dict[str, Any] | None:
    """The refusals about the portal piece itself, and about naming it elsewhere."""
    elsewhere = re.match(r"piece '[^']*' is a portal \(rules\.portals\) and may not (.*)$", text)
    if elsewhere:
        for start, where in _PORTAL_ELSEWHERE:
            if elsewhere.group(1).startswith(start):
                return {"says": _ONLY_FOR_PORTALS.format(where=where)}
    takes = re.match(r"piece '([^']*)' lists '[^']*' in captures, but a portal", text)
    if takes:
        where = f"{_piece_called(spec, takes.group(1))}'s 'may take only'"
        return {"says": _ONLY_FOR_PORTALS.format(where=where)}
    became = re.match(r"piece '([^']*)' is promoted_from '[^']*', a portal$", text)
    if became:
        where = f"{_piece_called(spec, became.group(1))}'s 'promoted from'"
        return {"says": _ONLY_FOR_PORTALS.format(where=where)}
    own = re.match(r"piece '([^']*)' \(a portal, rules\.portals\) (.*)$", text)
    if not own:
        clash = re.match(r"piece id (.*) is the name of a portal end beside rules\.portals", text)
        if clash:
            return {"says": f"The letter {clash.group(1)} is what a portal end is called. Give "
                    "that piece another letter."}
        return None
    rest = own.group(2)
    if re.search(r'must say mode "move"', rest):
        return {"says": 'Portals can\'t take anything, so every way a portal moves must be '
                '"move only".'}
    if "reads differently for the two players" in rest:
        return {"says": "Anyone may move a portal, so its moves must be the same both ways up."}
    if rest.startswith("has no moves, but rules.portals lets a player move"):
        rules = spec.get("rules") if isinstance(spec, dict) else None
        for group in portal_groups(rules):
            if group["piece"] != own.group(1) or group["spelling"] != "pair":
                continue
            index, pair = group["number"] - 1, _pair_view(group)
            for end in PORTAL_ENDS:
                if pair["moved_by"][end]:
                    who = _movers_words(pair["moved_by"][end], sides)
                    return {"pair": index + 1, "end": end, "says": f"Pair {index + 1}'s "
                            f"{_end_at(pair, end).removeprefix('the ')} may be moved by {who}, "
                            "but the portal piece "
                            'has no way to move. Give it a move, or set the end to "nobody".'}
        return {"says": "Somebody may move a portal, but the portal piece has no way to move."}
    if rest == "may not be royal":
        return {"says": "A portal belongs to nobody, so it can't be royal."}
    if rest.startswith("must have value 0"):
        return {"says": "A portal is nobody's piece, so it is worth 0."}
    if rest.startswith("may not declare"):
        return {"says": "A portal can't point a way, take anything or be promoted from another "
                f"piece: take '{rest.rsplit(' ', 1)[-1].replace('_', ' ')}' off the portal "
                "piece."}
    if rest.endswith("may not sweep"):
        return {"says": "A portal can't sweep: it moves only onto an empty square."}
    if "from_rank or region condition" in rest:
        return {"says": "A portal belongs to neither player, so its moves can't depend on a row "
                "counted from a player's side."}
    return None


def _off_board_words(got: str) -> str:
    """Where an end is said to be that is not a square: *isn't on a square of this board
    (z9)*, or, for an end not put down yet (a new card's, written as nothing), *isn't on
    the board yet* rather than the validator's *None* (Q995b)."""
    got = got.strip().strip(chr(39))
    if got in ("None", "null", ""):
        return "isn't on the board yet"
    return f"isn't on a square of this board ({got})"


def plain_problem(problem: str, spec: Any = None, sides: Any = None) -> str:
    """A validator problem in the designer's words: a portal's (D7) where it is one, two
    rules that can't go together (`_beside_problem`), a field of the whole army
    (`_army_problem`, Q995b), and `spec.plain`'s for every other."""
    said = portal_problem(problem, spec, sides)
    if said:
        return said["says"]
    words = (_beside_problem(problem, spec) or _army_problem(problem, spec)
             or _spec_plain_problem(problem, spec))
    # `spec.plain` names a rule where the validator wrote its path, and a path after an
    # article (*a rules.strike whose chain ...*) became *a the rule 'what a taken piece
    # becomes' whose chain*. The article goes.
    return re.sub(r"\ban? (the rule ')", r"\1", words)


# ------------------------------------------------------------------ two rules that can't meet

# The validator's sentence for a rule refused beside another (`rules.simultaneous beside
# rules.castling: castling moves two pieces in one move, ...`). `spec.plain` reads the two
# paths as two places run together (*The rule 'both players moving at once' beside the
# rule 'castling': ...*), which is the sentence a newcomer met three times on the way from
# chess to rounds (Q995a F2). Said here as one thing that can't go with another.
_BESIDE = re.compile(r"^rules\.(?P<rule>\w+) beside (?P<what>.+?): (?P<why>.+)$")


def _thing_words(what: str, spec: Any) -> str:
    """The other half of a *beside* sentence, as the cards call it."""
    win = re.fullmatch(r"rules\.win '(\w+)'( \(the default\))?", what)
    if win:
        return f"winning by {win[1].replace('_', ' ')}" + (", the default" if win[2] else "")
    valued = re.fullmatch(r"rules\.(\w+)(?:\.(\w+))? '([^']*)'", what)
    if valued:
        value = valued[3].replace("_", " ")
        if valued[2] is None:
            return f"the rule '{_rule_title(valued[1])}' set to '{value}'"
        words = field_words(f"rules.{valued[1]}.{valued[2]}")
        label = words[0] if words else valued[2].replace("_", " ")
        return f"'{label}' set to '{value}' in the rule '{_rule_title(valued[1])}'"
    field = re.fullmatch(r"rules\.(\w+)\.(\w+)", what)
    if field:
        words = field_words(f"rules.{field[1]}.{field[2]}")
        label = words[0] if words else field[2].replace("_", " ")
        return f"'{label}' in the rule '{_rule_title(field[1])}'"
    rule = re.fullmatch(r"rules\.(\w+)", what)
    if rule:
        return f"the rule '{_rule_title(rule[1])}'"
    # *a portal a player moves (moved_by)*: the key in brackets is the text's, not a word.
    said = _spec_plain_problem(re.sub(r" \([a-z]+_[a-z_]+\)", "", what), spec).rstrip(".")
    return said[:1].lower() + said[1:]


def _beside_problem(problem: str, spec: Any) -> str | None:
    """*The rule 'both players moving at once' can't go with the rule 'castling': castling moves two pieces in
    one move, and a round has one each.* The rule first, then what is in its way, then the
    validator's reason in plain words."""
    found = _BESIDE.match(problem)
    if found is None or found["rule"] == "portals":
        return None
    title = _rule_title(found["rule"])
    why = _spec_plain_problem(found["why"], spec)
    return (f"The rule '{title}' can't go with {_thing_words(found['what'], spec)}: "
            f"{why[:1].lower()}{why[1:]}")


# ------------------------------------------------------------------ a field of the whole army

# The validator checks some piece fields across the army at once and names them with no
# index (`pieces[].speed`, and `pieces.speed` in a version sentence). `spec.plain` reads a
# path into `pieces` only by its index, so these five sentences, about the newest fields,
# opened *Where the game comes from* and named neither the piece nor the field (Q995a F3).
# Here the pieces that carry the field are named from the draft.
_ARMY_PATH = re.compile(r"\bpieces(?:\[\])?\.(?P<key>[a-z_]+)\b")

# The three a newcomer meets most: a field of a game in rounds set on a game that is not
# in one. Each says where rounds are chosen.
_ROUNDS_ONLY = (
    (re.compile(r"^pieces\[\]\.speed orders the two moves of a round, and this game has none"),
     "speed", "a speed only matters in a game played in rounds", "clear {its} speed"),
    (re.compile(r"^pieces\[\]\.spy: a spy decides when a game plays in rounds"),
     "spy", "a spy only works in a game played in rounds", "take {its} spy away"),
    (re.compile(r"^pieces\[\d+\]\.beats says who wins a collision, and only matters"),
     "beats", "who a piece beats in a collision only matters in a game played in rounds",
     "clear who {it} beats"),
)
_CHOOSE_ROUNDS = "Choose 'in rounds' under 'how the players take turns', or"


def _carrying(spec: Any, key: str) -> list[int]:
    pieces = spec.get("pieces") if isinstance(spec, dict) else None
    if not isinstance(pieces, list):
        return []
    return [index for index, piece in enumerate(pieces)
            if isinstance(piece, dict) and key in piece]


def _pieces_called(spec: Any, indices: list[int]) -> str:
    """*Piece 2 (knight)*, *Pieces 2 (knight) and 3 (bishop)*: the place `spec.plain`
    gives one piece, for one or several."""
    pieces = spec["pieces"]

    def one(index: int) -> str:
        name = pieces[index].get("name")
        return f"{index + 1}" + (f" ({name})" if isinstance(name, str) and name.strip() else "")

    named = [one(index) for index in indices]
    if len(named) == 1:
        return f"Piece {named[0]}"
    return f"Pieces {', '.join(named[:-1])} and {named[-1]}"


def _army_problem(problem: str, spec: Any) -> str | None:
    """A problem about a field of the army, with the pieces that carry it named."""
    beats = re.match(r"^pieces\[(\d+)\]\.beats", problem)
    for pattern, key, says, way_out in _ROUNDS_ONLY:
        if not pattern.match(problem):
            continue
        carriers = [int(beats[1])] if key == "beats" and beats else _carrying(spec, key)
        if not carriers or not all(index < len(spec["pieces"]) for index in carriers):
            break
        several = len(carriers) > 1
        out = way_out.format(its="their" if several else "its", it="they" if several else "it")
        return f"{_pieces_called(spec, carriers)}: {says}. {_CHOOSE_ROUNDS} {out}."
    if _ARMY_PATH.search(problem) is None:
        return None
    plural: list[tuple[str, str]] = []

    def name(found: re.Match[str]) -> str:
        key = found["key"]
        words = field_words(f"pieces.{key}")
        label = words[0] if words else key.replace("_", " ")
        # The path the sentence is about, at its start or as what a version sentence says
        # the draft uses, is the pieces that carry it; one inside a reason is the field.
        subject = found.start() == 0 or problem[:found.start()].endswith("uses ")
        carriers = _carrying(spec, key) if subject else []
        if not carriers:
            return f"every piece's '{label}'" if subject else f"'{label}'"
        if len(carriers) > 1:
            plural.append((_pieces_called(spec, carriers[:1]), _pieces_called(spec, carriers)))
        return f"pieces[{carriers[0]}].{key}"

    said = _spec_plain_problem(_ARMY_PATH.sub(name, problem), spec)
    for alone, every in plural:
        said = said.replace(alone, every).replace(alone.lower(), every.lower())
    return said


def plain_problems(problems: list[str], spec: Any = None, sides: Any = None) -> list[str]:
    """`plain_problem` over a list, in order and one for one, except that two wild pieces'
    walks that overlap are one sentence per pair (`walk_overlaps`), where the validator
    says one per shared square."""
    out: list[str] = []
    overlaps_said = False
    for problem in problems:
        if _WALK_OVERLAP.match(problem) is None:
            out.append(plain_problem(problem, spec, sides))
        elif not overlaps_said:
            out.extend(walk_overlaps(problems))
            overlaps_said = True
    return out


# The validator's sentence for two wild pieces' walks on one square
# (`schema._check_npc_coherence`): one per square, with the reason the compiler's tables
# give. Sixteen of them for two wolves on one circuit, and a reason about tables, was the
# designer's refusal in the wolves walk (step 13, `docs/reference/designer-case-study-wolves.md`).
_WALK_OVERLAP = re.compile(
    r"^rules\.npcs\[(?P<later>\d+)\](?P<later_mirror> \(mirrored\))? and "
    r"rules\.npcs\[(?P<earlier>\d+)\](?P<earlier_mirror> \(mirrored\))? both walk "
    r"(?P<square>\S+?); routes may not share a square"
)


def walk_overlaps(problems: Iterable[str]) -> list[str]:
    """The overlapping walks among a draft's problems, one sentence per pair of walks:
    both wild pieces by the card's names (*wild pieces 2*, *wild pieces 1's mirror*), the
    squares they share, and the way out, *more on this walk*."""
    pairs: dict[tuple[tuple[int, bool], tuple[int, bool]], list[str]] = {}
    for problem in problems:
        found = _WALK_OVERLAP.match(problem)
        if found is None:
            continue
        earlier = (int(found["earlier"]), bool(found["earlier_mirror"]))
        later = (int(found["later"]), bool(found["later_mirror"]))
        squares = pairs.setdefault((earlier, later), [])
        if found["square"] not in squares:
            squares.append(found["square"])
    return [_overlap_sentence(earlier, later, squares)
            for (earlier, later), squares in pairs.items()]


def _overlap_sentence(earlier: tuple[int, bool], later: tuple[int, bool],
                      squares: list[str]) -> str:
    title = RULE_WORDS["npcs"][0]
    if not earlier[1] and not later[1]:
        both = f"{title.capitalize()} {earlier[0] + 1} and {later[0] + 1}"
    else:
        both = (f"{_walk_name(title, earlier).capitalize()} and "
                f"{_walk_name(title, later)}")
    if len(squares) == 1:
        where = squares[0]
    elif len(squares) <= 3:
        where = ", ".join(squares[:-1]) + " and " + squares[-1]
    else:
        where = f"{squares[0]} … {squares[-1]} ({len(squares)} squares)"
    mirror = earlier[1] or later[1]
    return (f"{both} both walk {where}. Two wild pieces can't share a square of their "
            "walks. To put a second one on the same walk, use 'more on this walk'"
            f"{' instead of the mirror' if mirror else ''}.")


def _walk_name(title: str, entry: tuple[int, bool]) -> str:
    index, mirrored = entry
    return f"{title} {index + 1}'s mirror" if mirrored else f"{title} {index + 1}"


def portal_errors(problems: list[str], spec: Any = None, sides: Any = None) -> list[dict[str, Any]]:
    """The portal refusals among a draft's problems, each with the pair and end it is about,
    for the pair's card and its *show me* (D7)."""
    out: list[dict[str, Any]] = []
    for problem in problems:
        said = portal_problem(problem, spec, sides)
        if said:
            out.append({key: said[key] for key in ("says", "pair", "end", "square", "place")
                        if key in said})
    return out


def _portals_answer(
    spec: Any, problems: list[str], preview: dict[str, Any] | None, engine: Any = None
) -> dict[str, Any]:
    """What the designer's portal row and cards are drawn from, beside a check's answer."""
    sides = (preview or {}).get("sides")
    return {
        "cards": portal_cards(spec, sides, engine),
        "errors": portal_errors(problems, spec, sides),
        "tool": portal_tool(spec),
    }


# ------------------------------------------------------------------ onto the hex lattice

# The hex lattice's own turn and mirror, in the axial coordinates every hex spec writes
# (`docs/reference/hex-build-plan.md`): a sixth of a turn sends the six neighbours round
# in order, (1, 0) (0, 1) (-1, 1) (-1, 0) (0, -1) (1, -1), and the mirror keeps forward,
# (0, 1), where it is and swaps the two cells beside it, (1, 0) and (-1, 1).
def _hex_turn(vector: tuple[int, int]) -> tuple[int, int]:
    x, y = vector
    return (-y, x + y)


def _hex_mirror(vector: tuple[int, int]) -> tuple[int, int]:
    x, y = vector
    return (-x, x + y)


def _hex_orbit(vectors: Iterable[Any], symmetry: str) -> list[list[int]]:
    """Each vector under the hex group that stands where the square's `symmetry` stood:
    `lateral` the mirror, `rotate4` the six turns, `dihedral8` the turns and the mirror.
    Each written vector, then its mirror, then their turns in order, without repeats."""
    out: list[list[int]] = []
    seen: set[tuple[int, int]] = set()

    def add(one: tuple[int, int]) -> None:
        if one not in seen:
            seen.add(one)
            out.append(list(one))

    for raw in vectors:
        start = (int(raw[0]), int(raw[1]))
        mirrors = [start]
        if symmetry in ("lateral", "dihedral8"):
            mirrors.append(_hex_mirror(start))
        for one in mirrors:
            add(one)
        if symmetry in ("rotate4", "dihedral8"):
            for one in mirrors:
                turned = one
                for _ in range(5):
                    turned = _hex_turn(turned)
                    add(turned)
    return out


def onto_hex(spec: Any) -> dict[str, Any]:
    """A draft whose cells were just made hexagons, rewritten so it can be a game (Q640).

    The validator refuses the square lattice's readings on a hex board, each a plausible
    wrong game: a move's `symmetry` (a quarter turn carries a hex vector off the lattice),
    the setup's rank mirror, a goal's rank mirror and a landing's symmetry. So choosing
    hexagons for any library game or template made a draft that was no game, said so in
    the validator's words, and offered no way back but rewriting every move by hand.

    Each is rewritten as its hex twin: a move keeps the vectors it wrote and takes the hex
    group in place of the square's (`_hex_orbit`, listed out under `none`), which on
    chess's own pieces gives Glinski's rook, bishop, knight and king; the armies face
    each other by the half turn; a goal turns with them. What has no twin, a `lame` leap
    or four facings, is left for the validator to name. Returns the draft and what was
    rewritten, in words; a draft not on the hex lattice comes back as it went.
    """
    import copy

    if not isinstance(spec, dict):
        raise DesignError("a draft")
    draft = copy.deepcopy(spec)
    board = draft.get("board")
    if not isinstance(board, dict) or board.get("lattice") != "hex":
        return {"spec": draft, "said": ""}

    def rewrite(move: Any) -> bool:
        if not isinstance(move, dict):
            return False
        symmetry = move.get("symmetry", "none")
        vectors = move.get("vectors")
        if symmetry not in ("lateral", "rotate4", "dihedral8") or not isinstance(vectors, list):
            return False
        if not all(isinstance(v, list) and len(v) == 2
                   and all(isinstance(n, int) for n in v) for v in vectors):
            return False
        move["vectors"] = _hex_orbit(vectors, symmetry)
        move["symmetry"] = "none"
        return True

    said = []
    moved = []
    for piece in draft.get("pieces") or []:
        if not isinstance(piece, dict) or not isinstance(piece.get("moves"), list):
            continue
        # Every move is rewritten, so no generator that stops at the first.
        if sum([rewrite(move) for move in piece["moves"]]):
            moved.append(str(piece.get("name") or piece.get("id") or "a piece"))
    if moved:
        said.append(f"the moves of the {', '.join(moved)} now go the hexagons' six ways")
    setup = draft.get("setup")
    if isinstance(setup, dict) and setup.get("orientation") != "rotate":
        setup["orientation"] = "rotate"
        said.append("the second army is the first turned half round")
    rules: dict[str, Any] = draft["rules"] if isinstance(draft.get("rules"), dict) else {}
    reach = rules.get("reach")
    if isinstance(reach, dict) and reach.get("mirror", True) not in ("rotate", False):
        reach["mirror"] = "rotate"
        said.append("the goal turns with the armies")
    for key in ("strike", "contacts"):
        landing = rules[key].get("landing") if isinstance(rules.get(key), dict) else None
        if isinstance(landing, dict) and rewrite(landing):
            said.append(f"the landing of '{_rule_title(key)}' goes the hexagons' six ways")
    if _rank_zones_named(draft):
        said.append("the promotion zone is named cell by cell")
    sentence = ("For hexagons, " + "; ".join(said) + ".") if said else ""
    return {"spec": draft, "said": sentence}


def _rank_zones_named(draft: dict[str, Any]) -> bool:
    """Each promotion zone spelled by rank rewritten as the cells it is, in place.

    On a hex lattice a zone is named squares only. The cells are the schema's own reading
    of the rank (`schema.promotion_squares`, the first player's half), so the zone is the
    one it was; a hexagon made of the board afterwards drops its absent corners from the
    list as it drops walls (`off_board`). True when anything was rewritten.
    """
    from chess_variant_sandbox.spec.schema import promotion_squares

    board = draft["board"]
    width, height = board.get("width"), board.get("height")
    rules = draft.get("rules")
    if not isinstance(rules, dict) or not isinstance(width, int) or not isinstance(height, int):
        return False
    promotion = rules.get("promotion")
    zones = promotion if isinstance(promotion, list) else [promotion]
    named = False
    for zone in zones:
        if not isinstance(zone, dict) or "squares" in zone:
            continue
        first, _ = promotion_squares(zone, width, height)
        if not first:
            continue
        zone.pop("rank", None)
        zone.pop("ranks", None)
        zone["squares"] = [_square_name(square, width) for square in sorted(first)]
        named = True
    return named


def _changes_from(parent: dict[str, Any] | None, spec: dict[str, Any]) -> list[str] | None:
    """What this draft changed from the game it was forked from, or None without one.

    A parent that no longer validates says nothing rather than failing the check: the
    draft is what is being asked about, and the parent is only its comparison.
    """
    if parent is None:
        return None
    try:
        return describe_changes(settled(parent), spec)
    except (SpecError, KeyError, TypeError, ValueError):
        return None


def refusals(spec: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Which of these rules this draft cannot take, and every reason why, in plain words.

    Option A9. The wild-piece refusal beside a strike is a good sentence and it arrived
    eight actions after the wolf was made (finding 2.10), so the menu asks for it before
    anything is added. One entry per candidate, empty where the rule would be accepted.

    **The page proposes and the validator judges.** The candidate *values* are the
    browser's, because what a card arrives holding is the form's answer and depends on
    the draft (`page/cards.CARD_DEFAULTS`); what those values are worth is asked here, of
    the same `validate_spec` every other rung asks. A draft that is not a game yet gets
    no refusals at all -- every rule would be refused for the draft's own errors, which
    is a menu greyed out from end to end and says nothing about any rule.

    **Every reason, at once, in the designer's words** (Q995b). This used to be the
    validator's first sentence, raw: from chess, rounds said only checkmate, and castling
    and taking in passing were each found by fixing the one before (Q995a F2). Each entry
    is now the whole list, through `plain_problems`.
    """
    import copy

    if not isinstance(spec, dict) or not isinstance(candidates, dict):
        raise DesignError("a draft and the rules to try on it")
    base = settled(spec)
    if _worth_saying(validate_spec(base)):
        return {"refused": {}, "asked": False}
    refused = {}
    for key, value in candidates.items():
        trial = copy.deepcopy(base)
        trial.setdefault("rules", {})[str(key)] = value
        problems = _worth_saying(validate_spec(settled(trial)))
        if problems:
            # Every reason once: several validator lines can read as one sentence.
            refused[str(key)] = list(dict.fromkeys(plain_problems(problems, trial)))
    return {"refused": refused, "asked": True}


# ------------------------------------------------------------------ how the players take turns

# The three ways, as the top of the designer offers them (Q995b): one after the other (no
# key), in rounds (`rules.simultaneous`), and more than one move a turn (`rules.turn`).
TURN_WAYS = ("alternate", "rounds", "several")


def _royal_names(trial: dict[str, Any]) -> str:
    names = [piece["name"] for piece in trial.get("pieces") or []
             if isinstance(piece, dict) and piece.get("royal") is True
             and isinstance(piece.get("name"), str) and piece["name"].strip()]
    return " or ".join(dict.fromkeys(names)) or "royal piece"


def _drop_rule(trial: dict[str, Any], key: str) -> bool:
    rules = trial.get("rules")
    return isinstance(rules, dict) and rules.pop(key, None) is not None


def _drop_in_rule(trial: dict[str, Any], key: str, inner: str) -> bool:
    rule = (trial.get("rules") or {}).get(key)
    return isinstance(rule, dict) and rule.pop(inner, None) is not None


def _drop_on_pieces(trial: dict[str, Any], key: str, only: int | None = None) -> list[str]:
    """The names of the pieces a field was taken off."""
    out = []
    for index, piece in enumerate(trial.get("pieces") or []):
        if isinstance(piece, dict) and key in piece and (only is None or only == index):
            del piece[key]
            out.append(str(piece.get("name") or piece.get("id")))
    return out


def _why(found: re.Match[str], spec: Any) -> str:
    said = _spec_plain_problem(found["why"], spec).rstrip(".")
    return said[:1].lower() + said[1:]


def _remedy(problem: str, trial: dict[str, Any]) -> str | None:
    """Make the one change that answers this problem, and say it; None where no change
    is ours to make.

    A change a turn-taking choice brings with it, and only those: a rule that cannot be
    played in rounds is taken out, a piece field that only means something in rounds is
    cleared. Which ones is the validator's to say -- this reads its sentences and never
    decides that a rule conflicts. A conflict with no change here (a sweep, a portal a
    player moves) stays a problem, and the choice is refused with it in plain words.
    """
    rounds = re.match(r"^rules\.simultaneous beside (?P<what>.+?): (?P<why>.+)$", problem)
    if rounds:
        what = rounds["what"]
        if re.fullmatch(r"rules\.win 'checkmate'( \(the default\))?", what):
            trial.setdefault("rules", {})["win"] = "royal_capture"
            return (f"Taking the {_royal_names(trial)} now wins, in place of checkmate: "
                    "there is no check when both players move at once.")
        dropped = re.fullmatch(
            r"rules\.(checkmate|forbid_check|check_limit|castling|en_passant)"
            r"|rules\.(royal_facing) 'forbidden'", what)
        if dropped:
            key = dropped[1] or dropped[2]
            if _drop_rule(trial, key):
                return f"The rule '{_rule_title(key)}' is taken out: {_why(rounds, trial)}."
            return None
        # Where a strike or a meeting puts its piece is not taken out: a chain needs the
        # piece to stay, so the change would trade one refusal for another, and the
        # person reads the first one, which is the reason.
        if what == "rules.drops.no_mate" and _drop_in_rule(trial, "drops", "no_mate"):
            words = field_words("rules.drops.no_mate")
            label = words[0] if words else "no mate"
            return (f"'{label}' is taken out of the rule '{_rule_title('drops')}': "
                    f"{_why(rounds, trial)}.")
        return None
    if problem.startswith("pieces[].speed orders the two moves of a round"):
        names = _drop_on_pieces(trial, "speed")
        return (f"The speed is cleared from the {_list_words(names)}: a speed only orders "
                "the two moves of a round.") if names else None
    if problem.startswith("pieces[].spy: a spy decides when a game plays in rounds"):
        names = _drop_on_pieces(trial, "spy")
        return (f"The {_list_words(names)} {'is' if len(names) == 1 else 'are'} no longer "
                "a spy: a spy only works in rounds.") if names else None
    beats = re.match(r"^pieces\[(\d+)\]\.beats says who wins a collision", problem)
    if beats:
        names = _drop_on_pieces(trial, "beats", int(beats[1]))
        return (f"Who the {names[0]} beats in a collision is cleared: it only matters when "
                "both players move at once.") if names else None
    passing = re.match(r"^rules\.en_passant beside a turn longer than one move", problem)
    if passing and _drop_rule(trial, "en_passant"):
        return (f"The rule '{_rule_title('en_passant')}' is taken out: in a turn of several moves a "
                "player could take their own pawn in passing.")
    return None


def take_turns(spec: Any, way: str, card: Any = None) -> dict[str, Any]:
    """The draft played another way, with every change that way needs made at once.

    The top of the designer offers how the players take turns (Q995b, from Q995a's F1 and
    F2): one after the other, in rounds, or more than one move a turn. From a copy of
    chess, rounds used to be three round trips: the rule was refused for checkmate, and
    then for castling, and then for taking in passing, each found alone. Here the choice
    is made, the validator is asked, and each problem it raises that a turn-taking choice
    answers is answered (`_remedy`), until none is left; what was changed comes back in
    sentences, so the page can say it and undo can take the whole step back.

    ``card`` is what the rule's card would arrive holding (the page's `fieldDefault`, as
    `refusals` takes it), used where the draft does not state the rule already.

    The answer is ``{"draft", "changes", "errors"}``. A choice that would leave a problem
    the draft did not have before is not made: ``draft`` is None and ``errors`` says every
    reason, in plain words. The draft is never changed in place.
    """
    import copy

    if not isinstance(spec, dict) or way not in TURN_WAYS:
        raise DesignError(f"a draft and one of {', '.join(TURN_WAYS)}")
    before = set(_worth_saying(validate_spec(settled(spec))))
    trial = copy.deepcopy(spec)
    rules = trial.setdefault("rules", {})
    if not isinstance(rules, dict):
        raise DesignError("a draft whose rules can be read")
    if way != "rounds":
        rules.pop("simultaneous", None)
    if way != "several":
        rules.pop("turn", None)
    if way == "rounds" and not isinstance(rules.get("simultaneous"), dict):
        rules["simultaneous"] = card if isinstance(card, dict) else {}
    if way == "several" and not isinstance(rules.get("turn"), dict):
        rules["turn"] = card if isinstance(card, dict) else {"moves": 2}

    changes: list[str] = []
    for _ in range(12):
        problems = _worth_saying(validate_spec(settled(trial)))
        said = None
        for problem in problems:
            if problem in before:
                continue
            said = _remedy(problem, trial)
            if said:
                changes.append(said)
                break
        if said is None:
            break
    left = [problem for problem in _worth_saying(validate_spec(settled(trial)))
            if problem not in before]
    if left:
        return {"draft": None, "changes": [], "errors": plain_problems(left, trial)}
    return {"draft": trial, "changes": changes, "errors": []}


def _errors_of(error: Exception, spec: Any = None) -> list[str]:
    """One message per problem, in the designer's words (`spec.plain`, W5b). The compiler
    raises them joined by newlines, and a list of eight things wrong with a draft is a
    list, not a paragraph."""
    if isinstance(error, SpecError):
        return plain_problems(list(error.errors), spec)
    text = str(error)
    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    return plain_problems(lines or [text], spec)


# Every sentence the version gates raise carries the stamp it wants, spelled this way.
_VERSION_STAMP = '"schema_version":'


def _worth_saying(problems: list[str]) -> list[str]:
    """The draft's problems, without the ones about a stamp the editor does not show.

    The form has no version control. `settled` raises the stamp for a person as soon as
    the rest of the draft is a game, so *declare "schema_version": N* standing beside a
    real error means only that the settling could not run yet -- and it appeared beside
    some other error in three of the four walks (finding 2.9). Alone it is the refusal
    itself, and then it stands.
    """
    real = [line for line in problems if _VERSION_STAMP not in line]
    return real or list(problems)


# The gate reports one rate for two counters, and walk 1.3 raised the wrong one twice
# before *open a game that did not finish* named the right one. Keyed by the core's own
# ending reason: the rule the counter is set by, and the form's label for that rule.
_CLOCK_COUNTERS = {
    "no_progress": ("no_progress_draw", "the no-progress rule"),
    "ply_limit": ("max_plies", "the turn limit"),
}
_CLOCK_REASON = STALLED


def _with_the_counter_named(reason: str, engine: Engine, spec: dict[str, Any]) -> str:
    """The stall sentence, with the counter that actually fired named and its setting.

    The gate keeps no per-game reasons, so the games are replayed under its own count,
    seed and budget -- which makes these the games it judged rather than a fresh sample.
    Threads never change the set, because each game draws its own stream. Only a draft
    that already stalls pays for the replay.
    """
    if _CLOCK_REASON not in reason:
        return reason
    import os

    from chess_variant_sandbox.loop.gates import SMOKE_GAMES

    try:
        threads = max(1, (os.cpu_count() or 2) - 1)
        played = engine.core.random_playouts(SMOKE_GAMES, 0, 20.0, threads)
    except Exception:  # noqa: BLE001 - a reason without its counter beats no reason
        return reason
    counted: dict[str, int] = {}
    for _plies, _winner, ending, _sampled, _samples in played:
        counted[ending] = counted.get(ending, 0) + 1
    rules = spec.get("rules") or {}
    fired = sorted(
        ((counted[key], key) for key in _CLOCK_COUNTERS if counted.get(key)), reverse=True
    )
    if not fired:
        return reason
    # In turns, a visitor's word for the counters' plies (D12): *120 were stopped by the
    # no-progress rule (now 40 turns) and 30 by the turn limit (now 200 turns)*.
    parts: list[str] = []
    for count, key in fired:
        rule, label = _CLOCK_COUNTERS[key]
        setting = rules.get(rule)
        parts.append(
            f"{count} {'were stopped ' if not parts else ''}by {label}"
            + (f" (now {setting} turns)" if isinstance(setting, int) else "")
        )
    return f"{reason}. Of {len(played)} games, {' and '.join(parts)}."


# ------------------------------------------------------------------ what a piece does


def move_map(
    spec: dict[str, Any], piece: int, square: int = -1, alone: bool = True
) -> dict[str, Any]:
    """Where one piece could go, and what it could take, from one square.

    The diagram every book of chess variants opens with, and the one thing a form full
    of ``{"kind": "leap", "vectors": [[1, 2]]}`` cannot tell you: whether what you wrote
    is the piece you meant.

    **It is not computed here.** Python builds positions and the engine answers them,
    which is the rule this whole package lives inside. Quiet moves come from asking the
    engine what is legal with the piece alone on an empty board. Captures cost one
    question per square -- put an enemy there, ask again, see whether the answer grew a
    move to it -- because "can it take on that square" is a different question from "can
    it move there", and six of the DSL's modes make them different answers. On a 12x8
    board that is 96 questions about a board with two pieces on it, which is under a
    millisecond and happens when somebody stops typing.

    ``alone`` picks which of two questions is being asked, and they are different
    questions. Alone on an empty board is what the *descriptor* means, which is the
    diagram; ``alone=False`` asks what the piece may do **from the opening position as it
    stands**, blockers, own pieces and every other rule included. There the answer is one
    call -- a capture is a legal move onto an occupied square, so the engine has already
    said which destinations are takes -- and the square defaults to the first one this
    kind of piece stands on rather than to the middle of the board.
    """
    engine = _draft_engine(spec)
    if not (0 <= piece < engine.spec.n_types):
        raise DesignError("That piece isn't in the game any more.")
    if not alone:
        return _live_map(engine, piece, square)
    return _alone_map(engine, piece, square)


def alone_maps(spec: dict[str, Any]) -> tuple[Engine, list[dict[str, Any]]]:
    """`move_map(spec, piece)` for every piece in turn, from one compile, with the engine
    that answered them.

    The library's piece maps (`library.piece_maps`) ask this of every game after a
    process's first listing; asked a piece at a time they compiled the rules once per
    piece, 661 compiles over the 87-game library where 87 do (Q890)."""
    engine = _draft_engine(spec)
    return engine, [_alone_map(engine, piece, -1) for piece in range(engine.spec.n_types)]


def _draft_engine(spec: dict[str, Any]) -> Engine:
    """The draft settled, validated and compiled, or the `DesignError` that says why not."""
    spec = settled(spec)
    # A draft the validator refuses is a refusal about the draft, a 400 with its first
    # sentences, and not the `SpecError` the compiler would raise into a 500 (Q541: a
    # palette king taken into an annihilation draft). The page already says why a draft
    # is not a game; this only keeps the answer an answer.
    problems = validate_spec(spec) if isinstance(spec, dict) else ["That draft could not be read."]
    if problems:
        raise DesignError("These rules aren't a complete game yet: "
                          + " ".join(plain_problems(problems[:2], spec)))
    # The compiler refuses a few things the validator passes (a wall set that bricks a
    # player in, a facing that turns to the same moves): the same kind of refusal about
    # the draft, so the same 400 and not a 500 (Q792, five of them on the live site on
    # 2026-10-05).
    try:
        return Engine(spec)
    except SpecError as error:
        raise DesignError(
            "These rules aren't a complete game yet: " + " ".join(_errors_of(error, spec)[:2])
        ) from error


def _alone_map(engine: Engine, piece: int, square: int) -> dict[str, Any]:
    """`move_map`'s answer with ``alone``: the piece by itself on an empty board."""
    compiled = engine.spec
    width, height = compiled.width, compiled.height
    base = engine.initial_state()
    # N5 (`docs/archive/plan-portals.md`, schema 19): the lone board keeps the portals where the
    # setup puts them, as it keeps the walls, so the engine's answer shows the transits.
    # Their board bytes are the opening position's own.
    # A fold (schema 22) is no board byte: it stays where the opening state's `folds` put it,
    # and a piece may be stood in it, so its square is no kept square (Q843d's fold preview).
    folded = compiled.portals.folds.starts if compiled.portals and compiled.portals.folds else ()
    kept = {at: base.board[at]
            for end, at in enumerate(compiled.portals.squares if compiled.portals else ())
            if not (end < len(folded) and folded[end] >= 0)}
    if square < 0 or square >= width * height:
        square = (height // 2) * width + width // 2
    if square in compiled.obstacles or square in kept:
        square = next(
            (s for s in range(width * height) if s not in compiled.obstacles and s not in kept),
            square,
        )

    from dataclasses import replace

    # The probe stands the piece up in its *initial* orientation, which is the one the
    # designer's board is showing and the only one a reader could name.
    mine = compiled.encode_piece(0, piece, compiled.facing_initial[piece])
    theirs = compiled.encode_piece(1, piece, compiled.facing_initial[piece])

    def board_with(*placements: tuple[int, int]) -> bytes:
        squares = bytearray(width * height)
        for where, what in [*kept.items(), *placements]:
            squares[where] = what
        return bytes(squares)

    bare = replace(base, board=board_with((square, mine)), side_to_move=0, unmoved=0)
    transits: dict[tuple[int, int], dict[str, Any]] = {}
    direct: set[int] = set()
    moves = set()
    for move in engine.legal_moves(bare):
        if move_from_square(move) == square:
            moves.add(move_to_square(move))
            _note_transit(engine, bare, move, transits, direct)

    takes: set[int] = set()
    for target in range(width * height):
        if target == square or target in compiled.obstacles or target in kept:
            continue
        with_enemy = replace(
            base,
            board=board_with((square, mine), (target, theirs)),
            side_to_move=0,
            unmoved=0,
        )
        for move in engine.legal_moves(with_enemy):
            if move_from_square(move) == square and move_to_square(move) == target:
                takes.add(target)
                _note_transit(engine, with_enemy, move, transits, direct)

    # Three sets, because the DSL makes them three different facts and a form full of
    # vectors shows none of them: a pawn may go forward and not take there, and may take
    # diagonally and not go there. `moves` and `takes` overlap for most pieces, which is
    # the ordinary case and is what the board draws as a plain destination.
    return {
        "from": square,
        # The board byte the probe stood up, so the editor can draw the map on an empty
        # board without working out an encoding for itself. Player-major and carrying the
        # piece's initial facing: the page would have to know both to reconstruct it, and
        # a second opinion about the board byte is how two drawings of one piece diverge.
        "value": mine,
        "width": width,
        "height": height,
        "moves": sorted(moves),
        "takes": sorted(takes),
        "moveOnly": sorted(moves - takes),
        "captureOnly": sorted(takes - moves),
        **_transits_payload(transits, direct),
        **walls_payload(compiled),
    }


def _note_transit(
    engine: Engine, state: Any, move: int,
    transits: dict[tuple[int, int], dict[str, Any]], direct: set[int],
) -> None:
    """One legal move of the piece asked about, sorted into a transit (the engine's own
    `portal_transit`: the end it goes in by, the end it comes out beside, the way out) or
    an arrival straight onto its square (D5)."""
    through = portal_transit(engine, state, move)
    to = move_to_square(move)
    if through is None:
        # Schema 22 (Q843d): a move through a fold is a transit too, its way in and out the
        # engine's (`fold_transit`); since schema 24 it ends where it comes out.
        folded = fold_transit(engine, state, move)
        if folded is None:
            direct.add(to)
            return
        transits.setdefault((to, folded["foldIn"]), {"to": to, **folded})
        return
    transits.setdefault((to, through["portal"]), {"to": to, **through})


def _transits_payload(
    transits: dict[tuple[int, int], dict[str, Any]], direct: set[int]
) -> dict[str, Any]:
    """The transits of a move map and the squares reached straight, where there are any
    transits; nothing at all in a game without portals, whose maps are as they were."""
    if not transits:
        return {}
    return {"transits": [transits[key] for key in sorted(transits)], "direct": sorted(direct)}


def _live_map(engine: Engine, piece: int, square: int) -> dict[str, Any]:
    """What this piece may do from the opening position, exactly as it stands.

    The other half of `move_map`, and the answer walk 1 of `docs/plan-designer-ux.md`
    kept wanting: a bishop's diagram is a star, and what a person is asking in the
    opening position is which two squares of it are left once its own pawns are there.
    One question to the engine -- the legal moves of the side to move -- and a capture is
    a legal move onto a square somebody else's piece stands on, so nothing here decides
    what a take is either.

    ``standing`` is what the caller asked about and ``from`` is what it got, because a
    piece may have been lifted off the board since the last answer.
    """
    compiled = engine.spec
    state = engine.initial_state()
    mover = state.side_to_move
    mine = [
        at
        for at in range(compiled.width * compiled.height)
        if (found := compiled.piece_at(state.board, at)) is not None
        and found == (mover, piece)
    ]
    standing = square if square in mine else (mine[0] if mine else -1)
    moves: set[int] = set()
    takes: set[int] = set()
    transits: dict[tuple[int, int], dict[str, Any]] = {}
    direct: set[int] = set()
    if standing >= 0:
        for move in engine.legal_moves(state):
            if move_from_square(move) != standing:
                continue
            target = move_to_square(move)
            moves.add(target)
            _note_transit(engine, state, move, transits, direct)
            here = compiled.piece_at(state.board, target)
            if here is not None and here[0] != mover:
                takes.add(target)
    return {
        **_transits_payload(transits, direct),
        "from": standing,
        "standing": sorted(mine),
        "live": True,
        "value": state.board[standing] if standing >= 0 else 0,
        "width": compiled.width,
        "height": compiled.height,
        "moves": sorted(moves),
        "takes": sorted(takes),
        "moveOnly": sorted(moves - takes),
        "captureOnly": sorted(takes - moves),
        **walls_payload(compiled),
    }


def move_from_square(move: int) -> int:
    from chess_variant_sandbox.engine.moves import MOVE_DROP, move_from, move_type

    return -1 if move_type(move) == MOVE_DROP else move_from(move)


def move_to_square(move: int) -> int:
    from chess_variant_sandbox.engine.moves import move_to

    return move_to(move)


# ------------------------------------------------------------------ teaching a move
#
# `docs/plan-designer-ux.md` stage 2, B2. A person clicks a square the piece should be
# able to reach and the designer answers with the readings of that click, each one a
# sentence and each one a descriptor the rule language can already express. Nobody types
# a vector.
#
# **Every square here is the engine's.** This module proposes descriptors -- that is
# arithmetic on the offsets that were clicked, and it decides nothing -- and then asks
# `engine/compile.py`, through a compiled spec, which squares each one reaches. A
# candidate whose answer does not cover what was clicked is dropped, two candidates whose
# answers are the same square set are one offer, and the sentence is `spec/describe.py`'s.
#
# The probe is a *stripped* spec: this board, one piece, and the rules that change where
# a vector lands -- the lattice, the walls and the seams. Not the draft itself, because
# compiling chess costs 29 ms and compiling the probe costs 3, and thirty candidates a
# click is the difference between an answer and a wait. What the descriptor means is a
# fact about the board's geometry, which the probe keeps whole; what the *piece* does
# under the rest of this game's rules is `move_map`'s answer, and the board goes back to
# asking it the moment an offer is taken.

# How many descriptors are probed for one click. Reached only by a click that is several
# squares out on a large board; the ordinary click proposes about thirty.
TEACH_LIMIT = 64

# The schema's `MAX_VECTOR_COMPONENT`: past it a leap cannot be written at all.
_MAX_LEAP = 8

# What the probe spec carries over from the draft, because each of them moves where a
# vector lands. Everything else about the game is left out on purpose.
_GEOMETRY_RULES = ("obstacles", "wrap")


def teach(
    spec: dict[str, Any],
    piece: int,
    square: int,
    targets: list[int],
    mean: str = "both",
    from_rank: bool = False,
) -> dict[str, Any]:
    """What a click on a square could mean, as sentences, cheapest reading first.

    ``square`` is where the piece stands and ``targets`` are the squares it should be
    able to reach. Each offer carries the descriptor, the engine's answer about it, and
    whether that answer is exactly what was clicked or more than it -- *one step that
    way* and *that step in all four diagonals* are both true of one click, and which one
    was meant is the person's to say.

    ``mean`` is what the click is for: ``both`` by default, because a piece takes where
    it moves unless somebody says otherwise (option A10). ``from_rank`` pins every
    reading to the rank the piece is standing on, which is a pawn's double step and
    changes no square, because the piece is standing on that rank while it is asked.
    """
    spec = settled(spec)
    compiled = _geometry(json.dumps(spec, sort_keys=True))
    if not (0 <= piece < compiled.n_types):
        raise DesignError("That piece isn't in the game any more.")
    width, height = compiled.width, compiled.height
    if not (0 <= square < width * height):
        raise DesignError("that piece is not standing anywhere on this board")
    if mean not in ("both", "move", "capture"):
        raise DesignError(f"a click means a move, a capture or both, not {mean!r}")
    wanted = sorted({at for at in targets if 0 <= at < width * height and at != square})
    if not wanted:
        return {"from": square, "width": width, "height": height, "offers": []}

    origin = (square % width, square // width)
    offsets = [(at % width - origin[0], at // width - origin[1]) for at in wanted]
    probe = _probe_spec(spec, compiled)
    # Checked once, and loudly. A probe the validator refuses answers every candidate
    # with nothing, which reads as *this game can say no such move* and is a lie about
    # the rule language.
    refused = validate_spec(_probed_with(probe, {"kind": "leap", "vectors": [[0, 1]]}))
    if refused:
        raise DesignError(
            "A move can't be taught on this board yet: " + "; ".join(refused[:2])
        )
    hexy = compiled.lattice == "hex"

    want = set(wanted)
    seen: dict[str, dict[str, Any]] = {}
    across = [_offsets_across(compiled.copies, origin, (at % width, at // width))
              for at in wanted]
    for move in _candidate_moves(offsets, compiled.lattice, across)[:TEACH_LIMIT]:
        reach = _probe_reach(probe, move, square)
        if not want <= reach:
            continue
        # A sweep takes everything it passes over, its own pieces included. Offered as a
        # reading of exactly the squares somebody clicked and never as a wider one:
        # every line has a sweep, and offering them all buries the readings a person
        # meant under a rule they have probably never heard of.
        if move["kind"] == "sweep" and reach != want:
            continue
        said = dict(move)
        if mean != "both":
            said["mode"] = mean
        if from_rank:
            said["condition"] = {"from_rank": origin[1]}
        # A reading nobody can read is worse than one reading fewer, so a descriptor
        # `describe` has no phrase for is dropped rather than offered as its vectors.
        sentence = plain_move_sentence(said, hexy)
        if sentence is None:
            continue
        # Kept by what it *says*: a leap of one square and a slide of one square are two
        # descriptors, reach the same square and are the same rule, and two identical
        # sentences in a menu is a reader wondering what the difference is. The cheaper
        # descriptor wins, which is the shorter spec.
        key = sentence
        offer = {
            "move": said,
            "says": sentence,
            # The engine's reach, read as what the click was for. A move-only descriptor
            # reaches those squares and never takes on them; a capture-only one takes
            # there and never moves. Which of the three it is was said by the person.
            "moves": [] if mean == "capture" else sorted(reach),
            "takes": [] if mean == "move" else sorted(reach),
            "exact": reach == want,
            "extra": sorted(reach - want),
            "cost": _descriptor_cost(said),
        }
        if key not in seen or offer["cost"] < seen[key]["cost"]:
            seen[key] = offer
    # The last key only splits a tie, and a tie of two exact readings is a seam: a step
    # of one across it and the long way round over the board (Q281c3). The short one is
    # what somebody drawing across a seam meant.
    offers = sorted(
        seen.values(),
        key=lambda one: (not one["exact"], len(one["extra"]), one["cost"],
                         sum(max(abs(dx), abs(dy)) for dx, dy in one["move"]["vectors"])),
    )
    return {"from": square, "width": width, "height": height, "offers": offers}


def _descriptor_cost(move: dict[str, Any]) -> int:
    """How much of the spec this descriptor is, counted as what a reader has to read.

    One for the descriptor, one per listed vector, one per optional field written. It
    ranks the offers, so the shortest way of saying what somebody drew is the one at the
    top -- the knight's eight squares as one vector turned eight ways rather than eight
    vectors. Two descriptors that read as the same sentence are one rule spelled two
    ways, and the shorter spelling is the one kept.
    """
    optional = ("symmetry", "range", "min_range", "mode", "condition")
    return 1 + len(move["vectors"]) + sum(1 for key in optional if key in move)


def _offsets_across(
    copies: tuple[Any, ...], origin: tuple[int, int], target: tuple[int, int]
) -> dict[str, tuple[int, int]]:
    """The offset to ``target`` through each copy of the board the first player may reach.

    A click is a raw difference of file and rank, which across a seam is the long way
    round or nothing. The piece that crosses reaches the target's image in the copy
    beyond the side it leaves by, and where that image is comes from
    `CompiledSpec.copies` (a cell's place in the copy minus its place on the board), so
    this adds a vector and works out no copy. Keyed by the side left by. Empty on a board
    with edges; a joined rectangle's copies are `CompiledSpec.grid_copies` (Q281c3),
    one past each joined edge and, on a torus, the four corners.

    The first player's permission, because the probe asks about the first player's
    piece. Whether an offset is short enough to state is the validator's to say (refusal
    7), and a candidate it refuses answers nothing.
    """
    out = {}
    for beyond in copies:
        if beyond.permission[0]:
            out[beyond.side] = (
                target[0] + beyond.vector[0] - origin[0],
                target[1] + beyond.vector[1] - origin[1],
            )
    return out


def _candidate_moves(
    offsets: list[tuple[int, int]],
    lattice: str,
    across: list[dict[str, tuple[int, int]]] | None = None,
) -> list[dict[str, Any]]:
    """Every descriptor worth asking the engine about for these offsets.

    Two vector sets: each clicked offset on its own, and all of them together. The first
    is what makes *demonstrate and generalise* work -- one click on a knight's square
    proposes that jump turned eight ways, which reaches seven squares nobody clicked --
    and the second is what closes a drawing that needs more than one direction.

    ``across`` is, per clicked offset, the same click reached through each copy of the
    board a piece may cross to (`_offsets_across`). Each of those is offered on its own,
    and with several clicks, all of them through the same side where that side reaches
    them, which is a drawing that runs over one seam. Empty on a board with no copies,
    so a board with edges gets exactly the candidates it always got.

    Each lattice takes its own symmetries (schema 18b): the hexagon's ``rotate6`` and
    ``dihedral12`` on a hex board, the square's three elsewhere; that is the schema's rule
    rather than this function's.
    """
    from math import gcd

    symmetries = HEX_SYMMETRIES if lattice == "hex" else SQUARE_SYMMETRIES
    groups: list[list[tuple[int, int]]] = [[one] for one in offsets]
    if len(offsets) > 1:
        groups.append(list(offsets))
    across = across or []
    groups.extend([one] for ways in across for one in ways.values())
    if len(offsets) > 1:
        for side in sorted({side for ways in across for side in ways}):
            over = [ways.get(side, direct) for ways, direct in zip(across, offsets, strict=True)]
            if over != list(offsets):
                groups.append(over)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def offer(move: dict[str, Any]) -> None:
        key = json.dumps(move, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(move)

    for vectors in groups:
        steps = [gcd(abs(dx), abs(dy)) for dx, dy in vectors]
        primitives = [
            [dx // step, dy // step] for (dx, dy), step in zip(vectors, steps, strict=True)
        ]
        near, far = min(steps), max(steps)
        listed = [[dx, dy] for dx, dy in vectors]
        leapable = all(abs(dx) <= _MAX_LEAP and abs(dy) <= _MAX_LEAP for dx, dy in vectors)
        for symmetry in symmetries:
            extra = {} if symmetry == "none" else {"symmetry": symmetry}
            if leapable:
                offer({"kind": "leap", "vectors": listed, **extra})
                # The same squares, with the one difference an empty board cannot show:
                # a piece standing in the way stops it.
                if far > 1:
                    offer({"kind": "lame", "vectors": listed, **extra})
            for kind in ("slide", "sweep"):
                offer({"kind": kind, "vectors": primitives, **extra})
                offer({"kind": kind, "vectors": primitives, **extra, "range": far})
                if far > 1:
                    offer({"kind": kind, "vectors": primitives, **extra,
                           "range": far, "min_range": near})
    # Cheapest first, so the limit cuts the long-winded readings rather than the short
    # ones, and fewer listed vectors wins a tie.
    return sorted(out, key=lambda move: (_descriptor_cost(move), len(move["vectors"])))


def _probe_spec(spec: dict[str, Any], compiled: Any) -> dict[str, Any]:
    """This board, one piece, and nothing else -- the cheapest spec that still compiles.

    It keeps the board's size and lattice and the rules that move where a vector lands.
    A placement is needed for the spec to validate at all; the probe replaces the board
    before it asks anything, so which free square it names does not matter.
    """
    width, height = compiled.width, compiled.height
    walls = set(compiled.obstacles)
    # A square whose images under either reflection are free too, because a mirrored
    # setup places the second army on one of them and the schema refuses a piece in a
    # wall. Square zero if the board has no such square; the spec is then refused and
    # `teach` says so rather than answering with nothing.
    def free_for_both(at: int) -> bool:
        file, rank = at % width, at // width
        flipped = (height - 1 - rank) * width + file
        turned = (height - 1 - rank) * width + (width - 1 - file)
        return not ({at, flipped, turned} & walls)

    free = next((at for at in range(width * height) if free_for_both(at)), 0)
    setup = spec.get("setup") or {}
    rules = spec.get("rules") or {}
    kept = {key: rules[key] for key in _GEOMETRY_RULES if key in rules}
    placed: dict[str, Any] = {
        "mirror": True,
        "placements": [{"piece": "A", "squares": [_square_name(free, width)]}],
    }
    # A hex board's two armies face each other by a half turn rather than by a rank
    # reflection, and the schema insists on it being written out.
    if setup.get("orientation"):
        placed["orientation"] = setup["orientation"]
    return {
        "schema_version": 1,
        "name": "probe",
        "description": "",
        "board": dict(spec.get("board") or {}),
        "pieces": [{"id": "A", "name": "probe", "value": 1, "moves": []}],
        "setup": placed,
        "rules": {"win": "annihilation", "stalemate": "draw", "max_plies": 40, **kept},
    }


def _square_name(square: int, width: int) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


def _probed_with(probe: dict[str, Any], move: dict[str, Any]) -> dict[str, Any]:
    """The probe spec with one descriptor in it, settled.

    The mode is left off on purpose: what is being asked is where the descriptor
    *goes*, and a capture-only move reaches nothing on a board with nobody to capture.
    """
    import copy

    asked = copy.deepcopy(probe)
    asked["pieces"][0]["moves"] = [
        {key: value for key, value in move.items() if key != "mode"}
    ]
    return settle_schema_version(asked)


@dataclass(frozen=True)
class _Geometry:
    """What `teach` reads off the compiled draft: its board, and nothing about its game."""

    width: int
    height: int
    lattice: str
    n_types: int
    obstacles: frozenset[int]
    copies: tuple[HexCopy, ...]


# Drafts whose geometry is kept. A person teaching a piece clicks the same draft over and
# over; the next edit is a new draft, and the few before it are what undo goes back to.
_GEOMETRY_KEPT = 8

# Probe answers kept: one descriptor, one board, one square. A second click on the same
# draft asks every candidate of the first click again.
_PROBES_KEPT = 4096


@lru_cache(maxsize=_GEOMETRY_KEPT)
def _geometry(settled_json: str) -> _Geometry:
    """The draft compiled once, for every click on it.

    **The draft's compile is the cost of a click on a joined hexagon**, not the probes.
    On a torus every line through a cell is the whole board, so Glinski's army compiles
    in seconds while a one-piece probe compiles in milliseconds; `teach` used to pay the
    first on every click. Keyed by the settled draft's text, so any edit is a new key.
    A draft that is not a game raises every time, since an exception is never kept.
    """
    try:
        compiled = Engine(json.loads(settled_json)).spec
    except (SpecError, ValueError) as error:
        raise DesignError(
            "these rules are not a game yet, so there is nothing to teach a move to: "
            + " ".join(_errors_of(error)[:2])
        ) from error
    return _Geometry(
        width=compiled.width,
        height=compiled.height,
        lattice=compiled.lattice,
        n_types=compiled.n_types,
        obstacles=frozenset(compiled.obstacles),
        # A hexagon's copies or a joined rectangle's (Q281c3): one of the two is empty.
        copies=tuple(compiled.copies) + tuple(compiled.grid_copies),
    )


def _probe_reach(probe: dict[str, Any], move: dict[str, Any], square: int) -> set[int]:
    """Which squares this descriptor reaches from ``square``, asked of the engine.

    One compile and one question, asked once: the answer is a fact about the probe, which
    is this board's geometry and one piece, so the same question gets the kept answer.
    """
    return set(
        _probe_reach_of(
            json.dumps(probe, sort_keys=True), json.dumps(move, sort_keys=True), square
        )
    )


@lru_cache(maxsize=_PROBES_KEPT)
def _probe_reach_of(probe_json: str, move_json: str, square: int) -> frozenset[int]:
    from dataclasses import replace

    probe, move = json.loads(probe_json), json.loads(move_json)
    try:
        engine = Engine(_probed_with(probe, move))
    except (SpecError, ValueError):
        return frozenset()
    compiled = engine.spec
    if square in compiled.obstacles:
        return frozenset()
    board = bytearray(compiled.width * compiled.height)
    board[square] = compiled.encode_piece(0, 0, compiled.facing_initial[0])
    state = replace(
        engine.initial_state(), board=bytes(board), side_to_move=0, unmoved=0
    )
    return frozenset(
        move_to_square(one)
        for one in engine.legal_moves(state)
        if move_from_square(one) == square
    )


# ------------------------------------------------------------------ showing a rule at work

# Games the core plays to find a rule at work, alternating greedy and random seats. A
# fixed count rather than a time budget, so the same draft always shows the same moves.
SHOW_GAMES = 32

# Plies shown before the one the rule changed.
SHOW_BEFORE = 2

# Random games played to find one that did not finish.
STUCK_GAMES = 16


def without_rule(spec: dict[str, Any], key: str) -> dict[str, Any]:
    """The draft with one rule taken out, settled the way the draft itself is."""
    import copy

    other = copy.deepcopy(spec)
    (other.get("rules") or {}).pop(key, None)
    return settled(other)


def demonstrate(spec: Any, key: str) -> dict[str, Any]:
    """Moves the engine played in which one rule changed the game.

    The core plays whole games of the draft. Each game is then walked in two engines,
    the draft and the draft without `key`, and the first ply where their answers differ
    is where the rule did something. The answers compared are the engine's own: whether
    the game is over, whether the move played is legal, the position it leaves, and
    which moves are legal. Nothing here decides a move or an outcome.

    A difference in the move actually played beats a difference in what could have been
    played, because it is visible on the board. The earliest over all games wins.
    """
    from chess_variant_sandbox.tournament.play import play_game

    if not isinstance(spec, dict):
        raise DesignError("That draft could not be read.")
    spec = settled(spec)
    if key not in (spec.get("rules") or {}):
        raise DesignError(f"this draft has no '{_rule_title(key)}' rule to show")
    other = without_rule(spec, key)
    if validate_spec(other):
        raise DesignError(
            f"without the '{_rule_title(key)}' rule this draft is not a game, so there is "
            "nothing to compare it with"
        )
    with_it, without_it = Engine(spec), Engine(other)

    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for seed in range(SHOW_GAMES):
        seats = ("greedy", "greedy") if seed % 2 == 0 else ("random", "random")
        moves = list(play_game(with_it, seats, seed, collect_series=False).moves)
        found = _first_difference(with_it, without_it, moves)
        if found is None:
            continue
        if found["visible"]:
            if best is None or found["at"] < best["at"]:
                best = found
        elif fallback is None or found["at"] < fallback["at"]:
            fallback = found
    chosen = best or fallback
    if chosen is None:
        raise DesignError(
            f"in {SHOW_GAMES} games the '{_rule_title(key)}' rule never changed a move, so there is "
            "nothing to show. It may say what the game does without it anyway"
        )
    return {
        "moves": chosen["moves"],
        "start": max(0, chosen["at"] - SHOW_BEFORE),
        "says": chosen["says"],
    }


def _first_difference(
    with_it: Engine, without_it: Engine, moves: list[int]
) -> dict[str, Any] | None:
    """The first ply where two engines disagree about one game, or None."""
    here, there = with_it.initial_state(), without_it.initial_state()
    if here.board != there.board:
        return {"at": 0, "visible": True, "moves": [],
                "says": "This rule changes the opening position."}
    quiet: dict[str, Any] | None = None
    for ply in range(len(moves) + 1):
        legal_here, legal_there = with_it.legal_moves(here), without_it.legal_moves(there)
        end_here = with_it.terminal(here, legal_here)
        end_there = without_it.terminal(there, legal_there)
        shown = moves[:ply]
        if _verdict(end_here) != _verdict(end_there):
            says = (
                f"This rule ends the game here, {_reason(end_here)}."
                if end_here is not None
                else f"Without this rule the game would end here, {_reason(end_there)}."
            )
            return {"at": ply, "visible": True, "moves": shown, "says": says}
        if end_here is not None or ply == len(moves):
            break
        move = moves[ply]
        text = with_it.describe_move(here, move)
        if move not in legal_there:
            return {"at": ply, "visible": True, "moves": moves[: ply + 1],
                    "says": f"{text} is only legal because of this rule."}
        after_here, after_there = with_it.apply(here, move), without_it.apply(there, move)
        if after_here.board != after_there.board:
            return {"at": ply, "visible": True, "moves": moves[: ply + 1],
                    "says": f"{text} leaves a different position because of this rule."}
        if quiet is None and set(legal_here) != set(legal_there):
            quiet = {"at": ply, "visible": False, "moves": shown,
                     "says": _legal_difference(with_it, without_it, here, there,
                                               legal_here, legal_there)}
        here, there = after_here, after_there
    return quiet


def _legal_difference(
    with_it: Engine,
    without_it: Engine,
    here: Any,
    there: Any,
    legal_here: list[int],
    legal_there: list[int],
) -> str:
    added = [m for m in legal_here if m not in set(legal_there)]
    if added:
        return (
            f"Here this rule allows {len(added)} move{'s' if len(added) > 1 else ''} that "
            f"would not be legal otherwise, such as {with_it.describe_move(here, added[0])}."
        )
    removed = [m for m in legal_there if m not in set(legal_here)]
    return (
        f"Here this rule rules out {len(removed)} move{'s' if len(removed) > 1 else ''} "
        f"that would be legal otherwise, such as "
        f"{without_it.describe_move(there, removed[0])}."
    )


def _verdict(outcome: Any) -> tuple[Any, Any] | None:
    return None if outcome is None else (outcome.winner, outcome.reason)


# The core's reasons for an ending (`Reason::as_str`), as the play board words them
# (`board.js` `REASON_WORDS`), in turns rather than moves (D45). A reason the core gains
# later reads *by the rules* until it is here: a reason's key is never shown.
_ENDING_WORDS = {
    "checkmate": "by checkmate", "stalemate": "by stalemate",
    "stalemate_loss": "by stalemate", "stalemate_win": "by stalemate",
    "royal_captured": "by capturing the royal piece",
    "royal_destroyed": "by destroying the royal piece",
    "annihilated": "by taking every piece", "repetition": "by repetition",
    "no_progress": "by the no-progress rule", "ply_limit": "by the turn limit",
    "reached": "by reaching the goal", "extinction": "by wiping out a whole kind of piece",
    "bare_royal": "by leaving the royal piece alone",
    "check_limit": "by giving check enough times", "coloured": "on ground held",
    "scored": "on points",
}


def _reason(outcome: Any) -> str:
    """How a game ended, as the end of a sentence: *by the turn limit*."""
    return _ENDING_WORDS.get(str(outcome.reason), "by the rules")


def _rule_title(key: str) -> str:
    """A rule by its card's own title (D45): *a capture must be taken*, never its key."""
    return RULE_WORDS[key][0] if key in RULE_WORDS else key.replace("_", " ")


def unfinished(spec: Any) -> dict[str, Any]:
    """A random game of the draft that did not finish, for the board to open at its end.

    The gate's own playouts keep no moves, so the core plays games of the same kind:
    random seats, fixed seeds. The first game stopped by the move limit or by no
    progress is chosen, then the first draw, then the shortest game when every game
    was won, since a quick win is the other way a gate fails.
    """
    from chess_variant_sandbox.tournament.play import play_game

    if not isinstance(spec, dict):
        raise DesignError("That draft could not be read.")
    spec = settled(spec)
    engine = Engine(spec)
    records = [
        play_game(engine, ("random", "random"), seed, collect_series=False)
        for seed in range(STUCK_GAMES)
    ]
    stalled = [r for r in records if r.reason in ("ply_limit", "no_progress")]
    drawn = [r for r in records if r.winner is None]
    if stalled:
        chosen = stalled[0]
        says = (f"This random game was stopped {_reason(chosen)} after "
                f"{chosen.plies} turns, with no winner.")
    elif drawn:
        chosen = drawn[0]
        says = (f"This random game was drawn {_reason(chosen)} after "
                f"{chosen.plies} turns.")
    else:
        chosen = min(records, key=lambda r: r.plies)
        says = (f"All {STUCK_GAMES} random games had a winner. This is the shortest, "
                f"won {_reason(chosen)} after {chosen.plies} turns.")
    return {
        "moves": list(chosen.moves),
        "start": len(chosen.moves),
        "says": says + " Step back to see how it got there.",
    }


def timeline(session: Any) -> list[dict[str, Any]]:
    """Every position a replayed session has been in, opening first.

    The board seeds its beat-stepped timeline from this, because a session that arrives
    already many plies in has no earlier snapshots to record.
    """
    states = [*session.states, session.state]
    comings = [coming_phases(session.engine, state) for state in states]
    return [
        {
            "board": list(state.board),
            "colour": list(state.colour),
            "hands": [list(state.hands[0]), list(state.hands[1])],
            "sideToMove": state.side_to_move,
            # Where each wild piece was going next from this position, and after that
            # (Q660, Q764), so the opening beat of a replayed game draws it as a live one does.
            "coming": [step.as_dict() for step in first],
            "comingThen": [step.as_dict() for step in then],
        }
        for state, (first, then) in zip(states, comings, strict=True)
    ]


# ------------------------------------------------------------------ remixing a game
#
# `docs/plan-designer-ux.md` option C and G6 of `docs/archive/plan-game-graph.md`. Two ways
# to start a draft from a game that already exists, and neither asks a person to know a
# field.
#
# **A design step** is one of `loop/operators.py`'s own edits, drawn under a fixed seed.
# The evolve loop makes thousands of these a night; what this adds is a way for a person
# to press one and see what came out. The words are the operator's own summary, written
# when it made the child, so the page never has to say what an edit was.
#
# **The game in between** is the other half. Two games two or three rules apart with
# nothing between them mark a game nobody has made, and where the rules that differ are
# rules the spec *declares*, carrying one of them across builds that game. Both are
# validated and compiled before they are offered, because a draft that cannot be opened
# is not a starting point.

# Seeds tried per design step before it counts as one that does not apply here. An
# operator picks its own target, so a draw can fail on a game that has nothing to remove
# and land on the next seed. Measured at Q350: on the committed corpus's chess, eight
# seeds reach 31 of the 33 operators in 0.04 s, and the two that never land are the two
# that need a rule the game does not have.
STEP_SEEDS = 8

# What `in_between` may carry across. A rule the spec declares is a value in one game and
# absent or different in the other, so moving it is a whole edit; a board, a piece or an
# opening position is not, and nothing here knows what half way between two boards is.
_RULE_UNIT = "rules."


def design_steps(
    spec: dict[str, Any], away_from: tuple[dict[str, Any], ...] = ()
) -> list[dict[str, Any]]:
    """Every design step that makes a different, valid game out of this one.

    One entry per operator that lands, carrying the operator's registry name, the seed it
    landed on and its own sentence about what it did. The caller presses one and `remix`
    draws it again under the same seed, which is the same child: an operator reads a
    `Random` it is handed and nothing else, so the draw is a pure function of the spec and
    the seed. Nothing here touches module-level `random`.

    A child that is the parent under another spelling is not a step. `identity_key` is
    the test rather than the hash, because a re-lettered or reordered spec is the same
    game and pressing it would be a change that changes nothing -- which is exactly what
    `save` refuses by name.

    ``away_from`` are games a step may not land on either (Q532). Several operators
    toggle -- one adds the blast, and pressed on the blasted game it takes it away -- so a
    step pressed twice walked the draft back to where it started, which is undo under
    another name and a surprise under this one. The editor passes the draft its last
    press was made on; a draw that lands there is a draw that did not land, and the
    operator tries its next seed.

    The children are validated and not compiled. Compiling thirty-one of them costs a
    second and a half, which is more than the press this answers is worth; `remix`
    compiles the one that is chosen.
    """
    from chess_variant_sandbox.loop.operators import OPERATORS
    from chess_variant_sandbox.spec import relate

    spec = settled(spec)
    mine = relate.identity_key(spec)
    refused = {mine} | {relate.identity_key(settled(other)) for other in away_from}
    found = []
    for operator in OPERATORS:
        for seed in range(STEP_SEEDS):
            drawn = _drawn(spec, operator, seed, refused)
            if drawn is None:
                continue
            found.append({"operator": operator.name, "seed": seed, "words": drawn[0]})
            break
    return found


def _drawn(
    spec: dict[str, Any], operator: Any, seed: int, refused: set[str]
) -> tuple[str, dict[str, Any]] | None:
    """One draw of one operator: its sentence and the child, or nothing at all.

    An operator that raises, declines or hands back something the validator refuses is a
    draw that did not land. None of the three is news: an operator picks its own target
    and most of them can fail to find one on a given game.
    """
    import copy
    import random

    from chess_variant_sandbox.spec import relate

    child = copy.deepcopy(spec)
    try:
        summary = operator.apply(child, random.Random(seed))
    except Exception:  # noqa: BLE001 - a draw that cannot apply is not an error
        return None
    if not summary:
        return None
    child = settle_schema_version(child)
    if validate_spec(child) or relate.identity_key(child) in refused:
        return None
    return str(summary), child


def remix(
    spec: dict[str, Any],
    operator: str,
    seed: int,
    name: str = "",
    rename: bool = True,
) -> dict[str, Any]:
    """One named design step, applied, as a draft the editor can open.

    The same draw `design_steps` listed, under the same seed, so what a person pressed is
    what they get. The draft carries the operator's own sentence as its description,
    because the editor's own header says only which game it started from, and what was
    changed is the whole of what a remix has to say.

    ``rename`` is off for a step pressed on a draft rather than on a saved game, and
    `draft_remix` is the caller that turns it off.
    """
    from chess_variant_sandbox.loop.operators import OPERATORS
    from chess_variant_sandbox.spec import relate

    spec = settled(spec)
    found = next((one for one in OPERATORS if one.name == operator), None)
    if found is None:
        raise DesignError(f"there is no design step called {operator}")
    drawn = _drawn(spec, found, int(seed), {relate.identity_key(spec)})
    if drawn is None:
        raise DesignError(
            "That change doesn't apply to this draft any more. Ask for the list again."
        )
    words, child = drawn
    parent = name or str(spec.get("name") or "this game")
    called = f"{parent} variant" if rename else str(spec.get("name") or parent)
    # The operator's sentence word for word, after the frame rather than inside it. It
    # often opens with a square or a piece's letter, and a sentence-case first letter
    # would rewrite the one thing here that is not this module's to write.
    note = (
        f"Made from {parent} in one design step. {words}."
        if rename
        else _one_step_further(str(spec.get("description") or ""), words)
    )
    return _draft(
        child,
        name=called,
        note=note,
        extra={"operator": operator, "seed": int(seed), "words": words},
    )


# The description a step on a saved game writes, read back so a further step on that
# draft can say it too. Only this module's own frame is read: a description a person wrote
# is theirs, and a step leaves it as it stands.
_STEPS_SAID = re.compile(r"Made from (?P<parent>.+?) in (?P<count>\w+) design steps?\. ", re.S)
_COUNTS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _one_step_further(description: str, words: str) -> str:
    """The description of a draft one design step on from ``description``.

    A draft remixed from a saved game carries *Made from X in one design step* and what
    that step did. A second step pressed on it used to keep that sentence as it was
    (Q532), so the draft described one change after a second had joined it, or walked it
    back. Where the description is still that frame, the new step is added to it and
    counted; anything else is somebody's own words and is kept.
    """
    said = _STEPS_SAID.match(description)
    if said is None:
        return description
    count = said["count"]
    if count in _COUNTS:
        before = _COUNTS.index(count) + 1
    elif count.isdigit():
        before = int(count)
    else:
        return description
    now = before + 1
    spelled = _COUNTS[now - 1] if now <= len(_COUNTS) else str(now)
    return (
        f"Made from {said['parent']} in {spelled} design steps. "
        f"{description[said.end():].rstrip()} {words}."
    )


# ------------------------------------------------------------------ the draft in hand
#
# `docs/plan-designer-ux.md` stage 4, C1 and C2. The two calls above answer about a game
# the library holds; these two answer about the draft somebody is editing, which is not in
# the library, usually has no hash anybody could name, and may not be a game yet.
#
# Nothing else differs, and neither of these draws a step of its own.


def draft_steps(spec: Any, away_from: Any = ()) -> dict[str, Any]:
    """Which design steps apply to a draft in hand, or why none is offered.

    ``away_from`` is the draft the last press was made on, when the page has one: a step
    that would land back on it is not offered (`design_steps`), since undo is that step.

    An operator edits a spec the validator accepts. On a draft it refuses, every draw
    fails and `design_steps` comes back empty -- and an empty list reads as *no step makes
    a different game out of this one*, which is a different sentence and a false one. So
    an unfinished draft is answered with the refusal instead, in the validator's own words
    and the same ones the verdict beside it is already showing.
    """
    if not isinstance(spec, dict):
        raise DesignError("That draft could not be read.")
    settled_spec = settled(spec)
    problems = _worth_saying(validate_spec(settled_spec))
    if problems:
        return {"steps": [], "why": plain_problem(problems[0], settled_spec)}
    back = tuple(one for one in (away_from or ()) if isinstance(one, dict))
    return {"steps": design_steps(settled_spec, back), "why": ""}


def draft_remix(spec: Any, operator: str, seed: int) -> dict[str, Any]:
    """One design step, pressed on the draft in hand.

    The child keeps the draft's own name and description. A step taken on a saved game
    makes a new game and is named after the game it came from; a step taken on a draft is
    that draft one step further on, and three presses would leave a draft called *chess
    variant variant variant*. What changed travels as `words` and the editor says it
    beside the button, where the press was.
    """
    if not isinstance(spec, dict):
        raise DesignError("That draft could not be read.")
    return remix(spec, operator, int(seed), rename=False)


def in_between(
    first: dict[str, Any],
    second: dict[str, Any],
    names: tuple[str, str] = ("", ""),
) -> dict[str, Any]:
    """The game between two others, or a sentence saying why there is none.

    A gap on the map is a pair two or three rules apart that no third game splits, which
    is a game nobody has made. Where every rule that differs is one the spec *declares*,
    that game can be built: take one of those rules from one end and write it into the
    other. The result is one rule from one end and the rest from the other, which is what
    *in between* has to mean for the link to be shorter on both sides.

    Four ways a candidate is refused, and each is checked here rather than hoped for.
    The validator may refuse it, because two rules can be legal apart and illegal
    together. `relate.identity_key` may say it is one of the two ends, which happens when
    the rule carried across was inert. The comparison may say it is no nearer either end
    than they already were. And the compiler may refuse what the validator allowed. A
    refusal is an answer -- `{"why": ...}` -- and never an error, because the map asks
    about every gap it draws and most of them have no game in between.

    The ends are read as they are written rather than `settled`, because `relate` does not
    compare a version stamp and settling one would be a change to somebody else's file.
    The child is settled, because carrying a rule across may be exactly what raises the
    stamp it needs.
    """
    from chess_variant_sandbox.spec import relate

    relation = relate.relate(first, second)
    if not relation.differences:
        return {"why": "These two are the same game, so there is nothing in between them."}
    if any(not one.unit.startswith(_RULE_UNIT) for one in relation.differences):
        return {
            "why": "These two differ in their board, pieces or starting position, not "
                   "only their rules, so there's no single game in between."
        }

    ends = (relate.identity_key(first), relate.identity_key(second))
    refused: list[str] = []
    for at, (base, other) in enumerate(((first, second), (second, first))):
        for difference in relation.differences:
            key = difference.unit.split(".", 1)[1]
            made = _rule_carried(base, other, key)
            if validate_spec(made):
                refused.append("invalid")
                continue
            if relate.identity_key(made) in ends:
                refused.append("an end")
                continue
            here = relate.relate(made, first).units
            there = relate.relate(made, second).units
            if here < 1 or there < 1 or here + there != relation.units:
                refused.append("no nearer")
                continue
            mine = names[at] or str(base.get("name") or "one of them")
            theirs = names[1 - at] or str(other.get("name") or "the other")
            return _draft(
                made,
                name=f"between {mine} and {theirs}",
                note=(
                    f"{mine} with {difference.sentence} taken from {theirs}. "
                    f"Nobody had made this game; it is {_rules(here)} from {mine} and "
                    f"{_rules(there)} from {theirs}."
                ),
                extra={
                    "words": difference.sentence,
                    "take": key,
                    "on": at,
                    "units": [here, there],
                },
                refusal=True,
            )
    return {"why": _no_candidate(refused)}


def _no_candidate(refused: list[str]) -> str:
    """Why no rule could be carried across, in the words of what went wrong."""
    if "invalid" in refused:
        return (
            "Carrying either rule across gives rules that don't fit together, so there "
            "is no game in between to offer."
        )
    if "an end" in refused:
        return (
            "Carrying either rule across gives one of these two games back rather than "
            "a third one, so there is nothing in between them after all."
        )
    return (
        "Carrying either rule across gives a game no nearer to either of these than "
        "they already are, so there is nothing in between to offer."
    )


def _rule_carried(
    base: dict[str, Any], other: dict[str, Any], key: str
) -> dict[str, Any]:
    """`base` with one `rules` key taken from `other`, through `spec/`'s own transforms.

    Absent in the other game means the rule goes, which is the same edit read the other
    way round and is how half of the corpus's gaps are crossed.
    """
    import copy

    from chess_variant_sandbox.spec.transform import set_rule, strip_rule

    made = copy.deepcopy(base)
    theirs = other.get("rules") or {}
    if key in theirs:
        set_rule(made, key, copy.deepcopy(theirs[key]))
    else:
        strip_rule(made, key)
    return settle_schema_version(made)


def _draft(
    child: dict[str, Any],
    name: str,
    note: str,
    extra: dict[str, Any],
    refusal: bool = False,
) -> dict[str, Any]:
    """A made game as a draft, compiled once so what is offered is a game that opens.

    The name and the description are written here rather than in the browser, so what a
    saved remix says about itself is the server's sentence and not a page's. Neither is
    inside the mechanics hash, so neither moves which game this is.

    The parent's provenance goes. It names the parent's own parents, and a draft that
    claimed them would be saying it came from somewhere it did not; `save` records the
    one parent it was forked from.
    """
    child = dict(child)
    child.pop("provenance", None)
    child["name"] = name
    child["description"] = note
    try:
        engine = Engine(child)
        moves = len(engine.legal_moves(engine.initial_state()))
    except (SpecError, ValueError) as error:
        if refusal:
            return {"why": "The game in between has rules that don't fit together: "
                           + " ".join(_errors_of(error, child)[:2])}
        raise DesignError(
            "That change would leave rules that don't fit together: "
            + " ".join(_errors_of(error, child)[:2])
        ) from error
    return {
        "spec": child,
        "name": name,
        "note": note,
        "hash": spec_hash(child),
        "openingMoves": moves,
        **extra,
    }


def _rules(count: int) -> str:
    return "one rule" if count == 1 else f"{count} rules"


# ------------------------------------------------------------------ keeping a draft


def save(
    spec: dict[str, Any],
    runs_dir: Path | str,
    parent: str = "",
    note: str = "",
    parent_spec: dict[str, Any] | None = None,
    parent_name: str = "",
    restate: bool = True,
) -> dict[str, Any]:
    """Write a hand-built game into the designed-games archive, as a candidate.

    Refuses an invalid spec, because a candidate that cannot be compiled is not a
    candidate -- the library would list it and every panel about it would fail.

    A game already known under this hash is returned rather than written again. The hash
    covers mechanics only, so re-saving a draft you have only renamed is not a new game,
    which is the same rule that stops a design run spending a tournament on a rename.

    **What a re-save may still change is the words** (Q595). A draft saved without a
    description used to keep that lack for ever: saving it again with one answered
    *Already saved* and wrote nothing. Now a re-save whose name or description differs
    from the kept one rewrites them in place (`Archive.restate`) and says which in
    `restated` -- the hash, and so the game, is unchanged. `restate=False` is the site's,
    for rules somebody else saved first or a game already published (`site/drafts.py`
    *What a re-save changes*); the words then stay, and `unchanged` names the ones the
    save asked for and did not get.

    **A fork that changed no rule is refused by name**, which is a different case from
    that one and used to be the worst behaviour in the editor. *Make one like this* on
    chess gives you `chess variant`; saving it unchanged wrote a candidate whose hash was
    chess's own, and `library.discover` resolves a hash collision in favour of the
    committed file -- so the page said *Saved as chess variant*, `created: true`, and
    *open its page* went to chess. The candidate was invisible for ever and nothing
    warned at any point. One comparison against the parent turns that into the clearest
    statement of what a game's identity is that the interface could make.

    The parent is compared *settled*, not as it sits on disk, because this function
    settles the draft: three committed rulesets declare schema 4 and validate under 3, so
    a fork of one of them settles to a hash the file does not have, and comparing raw
    hashes would let exactly the game this refusal exists for through.
    """
    spec = settled(spec)
    # `validate_spec` returns its problems rather than raising them, so a bare call is a
    # no-op that reads like a check. It let `{"name": "broken"}` into the archive, where
    # it became a library card whose every panel failed.
    problems = validate_spec(spec)
    if problems:
        raise DesignError(
            "These rules aren't a complete game yet: "
            + " ".join(plain_problems(problems[:4], spec))
            + (f" (and {len(problems) - 4} more)" if len(problems) > 4 else "")
        )
    digest = spec_hash(spec)
    if parent:
        forked = spec_hash(settled(parent_spec)) if parent_spec is not None else parent
        if forked == digest:
            # Never a hash (D27): a parent with no name is *the game you started from*.
            called = parent_name or "the game you started from"
            raise DesignError(
                f"Nothing you changed is a rule, so this is still {called}. A game is its "
                "rules: the board, the pieces, how they move, where they start and how "
                "it ends. Change something a player would notice and it becomes a new "
                "game."
            )
    archive = Archive(Path(runs_dir) / DESIGNED_RUN)
    existing = next((e for e in archive.entries if e.hash == digest), None)
    if existing is not None:
        kept = archive.load_spec(digest) or {}
        name = str(spec.get("name") or digest)
        description = str(spec.get("description") or "")
        differs = [
            word for word, was, now in (
                ("name", existing.name, name),
                ("description", str(kept.get("description") or ""), description),
            )
            if was != now
        ]
        if differs and restate:
            archive.restate(digest, name, description)
        return {"hash": digest, "name": name if differs and restate else existing.name,
                "created": False,
                "restated": differs if restate else [],
                "unchanged": [] if restate else differs,
                "path": str(Path(runs_dir) / DESIGNED_RUN / "specs" / f"{digest}.json")}

    archive.record(
        Entry(
            hash=digest,
            name=str(spec.get("name") or digest),
            generation=0,
            parent=parent or None,
            # "proposed" and not "evaluated": nothing has played it. The library shows it
            # as a candidate with no score, which is what it is.
            status="proposed",
            notes=note,
            # What tells `library.discover` this was built rather than found. A design
            # run's mutations and a person's drawing are both candidates in a run
            # directory and are not the same thing to a reader.
            detail={"origin": "designed"},
        ),
        spec,
    )
    return {"hash": digest, "name": str(spec.get("name") or digest), "created": True,
            "restated": [], "unchanged": [],
            "path": str(Path(runs_dir) / DESIGNED_RUN / "specs" / f"{digest}.json")}


# ------------------------------------------------------------------ building a player


def vocabulary() -> list[dict[str, Any]]:
    """The fifteen things an opponent can be told to care about, in English.

    Straight from ``agents/features.GLOSSES``, which is also what the proposing model is
    shown. One vocabulary, stated once: a slider labelled differently from the prompt
    would be a second opinion about what the term means.

    ``needs`` names the rule a term is inert without. Seven of the fifteen read a rule most
    rulesets do not declare, and a slider that silently does nothing is worse than one
    that is not offered -- so the editor greys them out and says why.
    """
    return [
        {"key": name, "gloss": GLOSSES[name], "needs": _NEEDS.get(name, "")}
        for name in VOCABULARY
    ]


# Which rule each term reads, where it reads one. Stated as the sentence the editor
# shows, not as a spec path: the point is to tell a person why a slider is inert.
_NEEDS = {
    "prey_weight": "pieces that may capture only certain other pieces",
    "territory_margin": "a rule that claims squares",
    "score_margin": "a rule that decides the game on a count",
    "score_urgency": "a rule that decides the game on a count",
    "check_margin": "a rule that ends the game after so many checks",
    "extinction_margin": "a rule that ends the game when a kind of piece runs out",
    "hand_material": "captured pieces returning to play",
    "promotion_occupancy": "promotion",
    "reach_progress": "a rule won by reaching a square",
    "facing_alignment": "pieces that may turn to face another way",
}


def live_terms(spec: dict[str, Any]) -> list[str]:
    """Which vocabulary terms are not inert on this ruleset.

    Read off the *compiled* spec rather than off the JSON, because that is what the
    evaluator reads: a rule stated one way in the file and another way by a player-keyed
    variant is one rule to the compiler and two shapes in the source. Sampling positions
    instead would be worse, not better -- `material` is a difference between the two
    sides and reads exactly zero from a symmetric opening, so a sampler would report the
    most important term in the vocabulary as dead on chess.
    """
    compiled = Engine(spec).spec
    has = {
        "prey_weight": bool(compiled.has_capture_matrix),
        "royal_exposure": bool(compiled.royal_types),
        "territory_margin": compiled.territory is not None,
        "score_margin": compiled.scoring is not None,
        "score_urgency": compiled.scoring is not None,
        # `check_limit` is an int and reads 0 where the rule is absent, not None -- so
        # this is the one of the ten that cannot be an `is not None` test, and writing it
        # that way marked the term live on every ruleset in the corpus.
        "check_margin": bool(compiled.check_limit),
        "extinction_margin": compiled.extinction is not None,
        "hand_material": compiled.drops is not None,
        "promotion_occupancy": bool(compiled.promotions),
        "reach_progress": compiled.reach is not None,
        # Schema 25: a facing no move turns is out of the term, so it alone keeps it dead.
        "facing_alignment": any(compiled.turns(t) for t in range(compiled.n_types)),
    }
    return [name for name in VOCABULARY if has.get(name, True)]


# Why a royal piece has no slider. The `_NEEDS` pattern applied to the other half of
# the panel: say why a control is inert rather than hiding it.
ROYAL_REASON = (
    "losing it ends the game, so it is worth everything and the evaluator prices it at "
    "nothing at all"
)


def piece_values(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """What each piece is worth to the rules, so a person can say it differently.

    The counterpart to ``vocabulary`` for the second half of an opponent. ``derived`` is
    the number the engine works out from the piece's average reach on an empty board
    blended with its declared value -- a fact about the rules, and the thing a slider
    moves *relative to*, because a table that runs from 1.25 to 74 on xiangqi has no
    absolute scale a person could hold in their head.

    ``royal`` is why the panel shows a disabled control rather than nothing. ``value``
    reaching a royal piece is the door back into what ``_reach_is_priceable`` measured,
    so it is refused everywhere and the interface says so in words.
    """
    compiled = Engine(spec).spec
    derived = Evaluator.for_spec(compiled).values
    art = icons_for(
        compiled.piece_names,
        compiled.piece_ids,
        (spec.get("appearance") or {}).get("icons"),
    )
    return [
        {
            "id": compiled.piece_ids[index],
            "name": compiled.piece_names[index],
            "icon": (art[index][0] or {}).get("href"),
            "derived": round(derived[index], 4),
            "royal": index in compiled.royal_types,
            "needs": ROYAL_REASON if index in compiled.royal_types else "",
        }
        for index in range(compiled.n_types)
    ]


def _absolute_values(
    spec: dict[str, Any], multipliers: list[float]
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Sliders as multipliers of the derived value, turned into a stored valuation.

    Three things happen here and each is a decision the plan argues for.

    **Multipliers in, absolute values out.** A stored opponent must not change its mind
    when ``_measured_values`` does; that is the whole point of content-addressing a
    player. So what the browser sends is "five times what the engine thinks" and what
    the file holds is a number.

    **All the same multiplier is no opinion at all.** The scale rule makes an override a
    *reallocation* of a fixed budget, so every slider at five is the same player as
    every slider at one -- which is the player with no override. Returning an empty
    table there rather than a copy of the derived one is what keeps that true of the
    *name* as well as of the play: the digest covers the values, so a table equal to the
    derived one would be a second name for one opponent. Detected before the arithmetic
    rather than after, so the claim does not rest on a float comparison.

    **A royal slider is not offered and not accepted.** Its multiplier is ignored rather
    than trusted, because the value is pinned at zero and the refusal downstream would
    name a control the person could not see.
    """
    compiled = Engine(spec).spec
    if len(multipliers) != compiled.n_types:
        raise DesignError(
            f"this game has {compiled.n_types} kinds of piece and {len(multipliers)} "
            "values were sent; the two have to line up"
        )
    derived = Evaluator.for_spec(compiled).values
    movable = [
        round(float(multipliers[index]), 4)
        for index in range(compiled.n_types)
        if index not in compiled.royal_types
    ]
    if not movable or len(set(movable)) == 1:
        return (), compiled.piece_ids

    absolute = tuple(
        0.0 if index in compiled.royal_types else derived[index] * float(multipliers[index])
        for index in range(compiled.n_types)
    )
    try:
        pair = PieceValues.of(derived, absolute, compiled.royal_types)
    except ValueError as error:
        raise DesignError(str(error)) from error
    return pair.objective, compiled.piece_ids


def build_opponent(
    ruleset_hash: str,
    name: str,
    plan: str,
    weights: list[float],
    values: list[float] | None = None,
    spec: dict[str, Any] | None = None,
    root: Path | str = STORE_DIR,
) -> dict[str, Any]:
    """Turn a set of sliders into a playable opponent, and keep it.

    The refusals are ``Heuristic``'s own: a vector of the wrong length, a weight past the
    bound, or every weight zero -- which is a player that moves at random and is a
    legitimate control but never something to offer as an opponent. Rejecting here rather
    than at play time is the whole value of the vector being data.

    ``values`` are multipliers of the derived piece values and need ``spec`` to become a
    valuation; see ``_absolute_values``. Without them the opponent has no opinion about
    the pieces, which is what every opponent built before stage 30C has, and it hashes
    exactly as it did.

    Returns the agent string, which is what the picker seats. Identity is the *vector
    pair*: two people who build the same opponent under two names build one opponent, and
    the second save is a rename of the first.
    """
    if not name.strip():
        raise DesignError("give it a name -- the picker has to call it something")
    piece_ids: tuple[str, ...] = ()
    valuation: tuple[float, ...] = ()
    if values:
        if spec is None:
            raise DesignError(
                "piece values are a multiplier of what this game's rules imply, so "
                "building one needs the ruleset in hand"
            )
        valuation, piece_ids = _absolute_values(spec, list(values))
    try:
        heuristic = Heuristic(
            name=name.strip(),
            plan=plan.strip(),
            weights=tuple(float(weight) for weight in weights),
            values=valuation,
        )
    except (HeuristicError, TypeError, ValueError) as error:
        raise DesignError(str(error)) from error

    store = Path(root) / CUSTOM_HEURISTICS
    # Read, not registered: a store that holds a clash from before Q433 still has every
    # opponent in it, and reading it as empty would overwrite them all.
    existing = _stored_set(ruleset_hash, store)
    kept = list(existing.heuristics) if existing else []
    _refuse_a_second_spelling(ruleset_hash, root, kept, heuristic)
    if not any(other.digest == heuristic.digest for other in kept):
        kept.append(heuristic)
    updated = HeuristicSet(
        ruleset_hash=ruleset_hash,
        heuristics=tuple(kept),
        model="built by hand",
        # Echoed only where something in the set prices the pieces, so a store that
        # outlived its ruleset says so instead of meaning something else.
        piece_ids=piece_ids or (existing.piece_ids if existing else ()),
    )
    updated.save(store / f"{ruleset_hash}.json")
    updated.register()
    from chess_variant_sandbox.web.opponents import HEURISTIC_NODES

    return {
        "agent": f"minimax:h={heuristic.digest},nodes={HEURISTIC_NODES}",
        "digest": heuristic.digest,
        "name": heuristic.name,
        "plan": heuristic.plan,
        "kept": len(kept),
    }


def _stored_set(ruleset_hash: str, store: Path) -> HeuristicSet | None:
    """One store as written, without registering it, or ``None`` if it cannot be read."""
    path = store_path(ruleset_hash, store)
    if not path.exists():
        return None
    try:
        return HeuristicSet.load(path)
    except (HeuristicError, OSError, ValueError):
        return None


def _refuse_a_second_spelling(
    ruleset_hash: str, root: Path | str, kept: list[Heuristic], heuristic: Heuristic
) -> None:
    """Refuse, before anything is written, an opponent the picker would refuse after.

    A digest is taken at unit length and the core plays the raw numbers, so one digest
    spelled twice is two players under one name, and registration refuses the second
    (Q433). The picker registers this game's written and proposed stores before
    `custom/`, whether or not this process has listed them yet, and refuses a clashing
    store whole -- so everything the updated store would hold is checked against those
    stores, against what this process already holds, and against itself.
    """
    from chess_variant_sandbox.web.opponents import SOURCES

    seen: list[Heuristic] = []
    for subdir, _group, _anchored in SOURCES:
        if subdir != CUSTOM_HEURISTICS:
            found = _stored_set(ruleset_hash, Path(root) / subdir if subdir else Path(root))
            seen.extend(found.heuristics if found else ())
    for candidate in (*kept, heuristic):
        held = other_spelling(candidate, seen)
        if held is not None:
            whose = (
                f"{candidate.name!r}" if candidate is heuristic
                else f"{candidate.name!r}, built here before,"
            )
            raise DesignError(
                f"{whose} has the balance of {held.name!r} with other exact numbers; "
                "the engine plays the exact numbers, so the one name would call two "
                "players. Use the numbers of the one already offered, or change the balance"
            )
        seen.append(candidate)


def forget_opponent(
    ruleset_hash: str, digest: str, root: Path | str = STORE_DIR
) -> dict[str, Any]:
    """Drop one hand-built opponent. Only ever from the custom store.

    Read whole and never registered (`_stored_set`, the way `for_game` reads a store
    now, Q446) rather than through `load_set`. `load_set` registers every entry in one
    call, all or nothing, and a `custom/` store saved before Q433 can hold two spellings
    of one digest -- so forgetting an unrelated entry in such a store raised instead of
    forgetting anything (Q449).

    Matched by digest and removed **by position**, the first entry that spells it: a
    clash is two entries whose `digest` computes the same from different raw numbers,
    so dropping every entry equal to the digest would silently take both. Removing one
    position leaves everything else -- the other spelling included -- exactly where it
    stood in the file.
    """
    store = Path(root) / CUSTOM_HEURISTICS
    found = _stored_set(ruleset_hash, store)
    if found is None:
        raise DesignError("nothing has been built for this game")
    heuristics = list(found.heuristics)
    at = next((index for index, h in enumerate(heuristics) if h.digest == digest), None)
    if at is None:
        raise DesignError(f"no opponent {digest} was built here")
    kept = heuristics[:at] + heuristics[at + 1 :]
    path = store / f"{ruleset_hash}.json"
    if kept:
        HeuristicSet(
            ruleset_hash=ruleset_hash, heuristics=tuple(kept), model="built by hand"
        ).save(path)
    else:
        path.unlink(missing_ok=True)
    return {"kept": len(kept)}


# ------------------------------------------------------------------ uploaded icons


def icon_data_uri(svg: str) -> str:
    """An uploaded SVG as a data URI a spec's ``appearance.icons`` can name.

    ``appearance.icons`` maps a piece id to a *string*, which the schema requires to be
    non-empty and is otherwise free -- so an inlined image validates today and needed no
    schema change. ``viz/pieces.icon_for`` recognises the prefix and returns it unchanged.

    The refusals are about the file being what it claims. A data URI goes into an
    ``<image href>`` inside the board's SVG, where a browser renders it in a separate
    document: it cannot reach the page, but it can be enormous, and a spec is a file
    somebody reads.
    """
    import base64
    import re

    text = svg.strip()
    if text.startswith("data:image/svg+xml"):
        return text
    if "<svg" not in text[:400].lower():
        raise DesignError("that does not look like an SVG file")
    if len(text) > 64_000:
        raise DesignError(
            f"that SVG is {len(text) // 1024}kB; the spec has to stay a file somebody "
            "can read, so keep icons under 64kB"
        )
    # A `<script>` in an icon never runs -- an SVG in an `<image>` is a separate,
    # inert document -- but it also has no business in a piece drawing, and a spec is
    # committed and read by people. Refused rather than stripped: quietly editing
    # somebody's file is worse than declining it.
    if re.search(r"<\s*script", text, re.I):
        raise DesignError("that SVG contains a script; icons are pictures only")
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return "data:image/svg+xml;base64," + encoded


def blank_spec() -> dict[str, Any]:
    """The shape the editor starts from when nothing is chosen. Not a template: this is
    what "reset" means, and it is deliberately the first template rather than an empty
    object, because an empty object is eight validation errors."""
    return dict(templates()[0]["spec"])


def schema_note() -> dict[str, Any]:
    """What the editor may build, so it can say so rather than fail late."""
    return {
        "version": SCHEMA_VERSION,
        "maxWeight": MAX_ABS_WEIGHT,
        "terms": len(VOCABULARY),
        # The widest a valuation may be, which is the validator's own cap on piece
        # types. How many *this* game has comes with the game, because a value vector is
        # ruleset-shaped where the weight vector is not.
        "maxPieces": MAX_PIECE_TYPES,
        "valueRange": [VALUE_MULTIPLIER_MIN, VALUE_MULTIPLIER_MAX],
    }


# ------------------------------------------------------------------ a rule file, loaded


RULE_FILE_BYTES = 32 * 1024
"""The largest rule file the designer reads, in bytes of UTF-8 (Q830b).

The largest game in the library is about 10 KB (a shogi with drops), so this is three of
those. It also keeps a file, sent as a string inside a design POST's JSON, under the
site's `quota.DESIGN_BODY_BYTES` however its quotes and line ends are escaped in transit
-- so the refusal a person reads is this one, about their file, and not the body cap's.
"""

RULE_FILE_DEPTH = 16
"""How deeply a rule file's objects and lists may nest (Q830b). The deepest game in the
library nests 7; the rule language never needs more than about that. Counted before the
text is parsed, so a file of ten thousand brackets is refused without being built."""


def read_rule_file(text: Any) -> dict[str, Any]:
    """A rule file somebody wrote, by hand or with their own AI, read as a draft (Q830b).

    The text of the file or of the *game as text* box, and nothing else: the page sends
    what it was given and decides nothing about it. The answer is ``valid`` with the
    draft, settled as `check` settles one, or not valid with every problem, and then
    the page keeps the draft it had.

    **The problems are the validator's own lines**, the ones `validate` prints and
    `RULES.md` shows under *How a wrong file is refused*, not `plain_problems`' form
    words: whoever wrote the file reads it by its paths (``pieces[1].moves[0].kind``), and
    the form those words describe is not the thing they wrote. The same two passes as the
    designer's check -- the validator, then the compiler -- because a file the validator
    accepts and the compiler refuses is not a game either.

    Nothing is kept and nothing is played. Reading is milliseconds; a save, with every
    limit and check the site puts on one, is still the only way a draft is written down.
    """
    if not isinstance(text, str):
        return _not_loaded(["That file could not be read as text."])
    # A byte-order mark is how some editors start a UTF-8 file; it is not part of the JSON.
    text = text.removeprefix("﻿")
    if len(text.encode("utf-8", "replace")) > RULE_FILE_BYTES:
        return _not_loaded([
            f"That is more than {RULE_FILE_BYTES // 1024} KB, and a rule file may be at most "
            f"{RULE_FILE_BYTES // 1024} KB. The largest game in the library is about 10 KB."
        ])
    if not text.strip():
        return _not_loaded(["That file is empty."])
    deepest = _nesting(text)
    if deepest > RULE_FILE_DEPTH:
        return _not_loaded([
            f"That nests {deepest} levels deep, and a rule file may nest at most "
            f"{RULE_FILE_DEPTH}. No game needs more than about 7."
        ])
    try:
        found = json.loads(text)
    except json.JSONDecodeError as error:
        return _not_loaded([
            f"That is not JSON: {error.msg} at line {error.lineno}, column {error.colno}."
        ])
    if not isinstance(found, dict):
        kind = "a list" if isinstance(found, list) else "a single value"
        return _not_loaded([
            f"A rule file is one JSON object, {{ ... }}, and this is {kind}."
        ])
    spec = settled(found)
    problems = _worth_saying(validate_spec(spec))
    if problems:
        return _not_loaded(problems)
    try:
        engine = Engine(spec)
        engine.legal_moves(engine.initial_state())
    except SpecError as error:
        return _not_loaded(list(error.errors))
    except ValueError as error:
        lines = [line.strip(" -") for line in str(error).splitlines() if line.strip()]
        return _not_loaded(lines or [str(error)])
    return {"valid": True, "errors": [], "spec": spec}


def rule_file_kit() -> dict[str, Any]:
    """What the page needs to offer a rule file: the link to the published rule language,
    the two controls' labels `RULES.md` tells a reader to use, and how much of a file to
    send (one byte over the limit is enough for this module to refuse it in its words)."""
    return {
        "rulesUrl": reference.PUBLISHED,
        "load": reference.LOAD_CONTROL,
        "paste": reference.PASTE_CONTROL,
        "bytes": RULE_FILE_BYTES,
    }


def _not_loaded(errors: list[str]) -> dict[str, Any]:
    return {"valid": False, "errors": errors}


def _nesting(text: str) -> int:
    """How deeply the text's objects and lists nest, read off its brackets.

    Brackets inside strings are not counted, so an icon's or a description's text cannot
    trip it. Malformed JSON gets an answer too -- the parser that runs next words that
    refusal -- because all this needs is that no file is parsed deeper than it allows.
    """
    depth = deepest = 0
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif char in "]}":
            depth -= 1
    return deepest
