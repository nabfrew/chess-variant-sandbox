"""The validator's refusals, one test each (`docs/plan-tests.md` §8 F8, T2).

`spec/schema.py` is the trust boundary for a proposed rule (CLAUDE.md, rule 1): a spec that
passes `validate_spec` is one the compiler consumes without checks of its own. The audit
found most of its refusal branches reached by no test, so a refusal that stopped refusing
would have failed nothing. Each case here is an inline spec built to trip one refusal,
and asserts it by the words a designer reads, not merely that the spec fails.

Every spec is built from `BASE` (or `PORTALS`, for schema 19's checks), a small game that
validates clean, with one change. Nothing here reads a committed ruleset.

`UNHASHABLE` holds the malformed values F8 found the validator raising on instead of
refusing (strict xfails until Q825 fixed the validator); each now draws its refusal.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.schema import (
    ENDING_RESULTS,
    FACING_NAMES,
    MAX_BOARD_DIM,
    MAX_MOVE_DESCRIPTORS,
    MAX_PIECE_TYPES,
    MAX_STOCK,
    MODES,
    ORIENTATIONS,
    SCHEMA_VERSION,
    STOCK_UNLIMITED,
    npc_routes,
    parse_file,
)

STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}

# Two kings and two rooks on a 5 by 5 board: small, legal, and naming no game.
BASE: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "name": "refusal base",
    "board": {"width": 5, "height": 5},
    "pieces": [
        {"id": "K", "royal": True, "moves": [STEP]},
        {"id": "R", "value": 5, "moves": [ROOK]},
    ],
    "setup": {
        "mirror": True,
        "placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1"]},
        ],
    },
    "rules": {"win": "checkmate"},
}

# BASE with one fixed pair of portals on the empty middle rank (schema 19).
PORTALS: dict[str, Any] = copy.deepcopy(BASE)
PORTALS["pieces"].append({"id": "O", "value": 0, "moves": []})
PORTALS["rules"]["portals"] = [{"piece": "O", "orange": "b3", "blue": "d3"}]

DROP = object()  # a patch value that removes the key


def _patched(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """``base`` with each dotted path set (a number indexes a list), or removed."""
    spec = copy.deepcopy(base)
    for path, value in changes.items():
        *parents, last = path.split(".")
        node: Any = spec
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
        key: Any = int(last) if isinstance(node, list) else last
        if value is DROP:
            del node[key]
        elif isinstance(node, list) and key == len(node):
            node.append(copy.deepcopy(value))
        else:
            node[key] = copy.deepcopy(value)
    return spec


def test_the_bases_are_valid() -> None:
    assert validate_spec(BASE) == []
    assert validate_spec(PORTALS) == []


def test_a_spec_that_is_not_an_object_is_refused_by_name() -> None:
    assert validate_spec(["not", "a", "spec"]) == ["spec must be a JSON object"]


# Each row: the change to BASE, and the refusal it must draw, word for word.
SHAPE = [
    # ---------------------------------------------------------------- board
    pytest.param({"board.width": "5"}, "board.width must be an integer", id="board-width-type"),
    # ---------------------------------------------------------------- pieces
    pytest.param(
        {"pieces": [{"id": f"P{i}", "moves": [STEP]} for i in range(MAX_PIECE_TYPES + 1)]},
        f"at most {MAX_PIECE_TYPES} piece types, got {MAX_PIECE_TYPES + 1}",
        id="too-many-types",
    ),
    pytest.param({"pieces.2": "Q"}, "pieces[2] must be an object", id="piece-not-object"),
    pytest.param(
        {"pieces.1.id": "ROO"}, "pieces[1].id must be a string of 1-2 characters", id="piece-id"
    ),
    pytest.param({"pieces.1.name": 7}, "pieces[1].name must be a string", id="piece-name"),
    pytest.param({"pieces.1.royal": "yes"}, "pieces[1].royal must be a boolean", id="royal"),
    pytest.param({"pieces.1.value": "5"}, "pieces[1].value must be a number", id="value-type"),
    pytest.param(
        {"pieces.1.value": 101}, "pieces[1].value must be between 0 and 100", id="value-range"
    ),
    pytest.param(
        {"pieces.1.promoted_from": 3}, "pieces[1].promoted_from must be a piece id",
        id="promoted-from",
    ),
    # ---------------------------------------------------------------- facing
    pytest.param({"pieces.1.facing": 2}, "pieces[1].facing must be an object", id="facing-type"),
    pytest.param(
        {"pieces.1.facing": {"axes": 3}}, "pieces[1].facing.axes must be one of 2, 4",
        id="facing-axes",
    ),
    pytest.param(
        {"pieces.1.facing": {"axes": 2, "initial": "up"}},
        f"pieces[1].facing.initial must be one of {', '.join(FACING_NAMES)}",
        id="facing-initial",
    ),
    # ---------------------------------------------------------------- captures
    pytest.param(
        {"pieces.1.captures": "K"}, "pieces[1].captures must be an array of piece ids",
        id="captures-type",
    ),
    pytest.param(
        {"pieces.1.captures": [1]}, "pieces[1].captures entry 1 must be a piece id",
        id="captures-entry",
    ),
    pytest.param(
        {"pieces.1.captures": ["K", "K"]}, "pieces[1].captures repeats a piece id",
        id="captures-repeat",
    ),
    # ---------------------------------------------------------------- moves
    pytest.param(
        {"pieces.1.moves": [ROOK] * (MAX_MOVE_DESCRIPTORS + 1)},
        f"pieces[1].moves may hold at most {MAX_MOVE_DESCRIPTORS} descriptors",
        id="too-many-moves",
    ),
    pytest.param({"pieces.1.moves": [1]}, "pieces[1].moves[0] must be an object", id="move-type"),
    pytest.param(
        {"pieces.1.moves.0.mode": "fly"},
        f"pieces[1].moves[0].mode must be one of {MODES}, got 'fly'",
        id="move-mode",
    ),
    pytest.param(
        {"pieces.1.moves.0.screen": "anyone"},
        "pieces[1].moves[0].screen must be one of ('any', 'own', 'enemy'), got 'anyone'",
        id="screen-owner",
    ),
    pytest.param(
        {"pieces.1.moves.0.screen": "any"},
        "pieces[1].moves[0].screen only means something on a hop, not a slide",
        id="screen-not-hop",
    ),
    pytest.param(
        {"pieces.1.moves.0.range": -1}, "pieces[1].moves[0].range must be a non-negative integer",
        id="range-type",
    ),
    pytest.param(
        {"pieces.1.moves.0.range": 99}, "pieces[1].moves[0].range may not exceed 16",
        id="range-max",
    ),
    pytest.param(
        {"pieces.1.moves.0.min_range": 0}, "pieces[1].moves[0].min_range must be a positive integer",
        id="min-range",
    ),
    pytest.param(
        {"pieces.1.moves.0": {"kind": "hop", "vectors": [[1, 0]], "screen_range": -1}},
        "pieces[1].moves[0].screen_range must be a non-negative integer",
        id="screen-range-type",
    ),
    pytest.param(
        {"pieces.1.moves.0": {"kind": "hop", "vectors": [[1, 0]], "screen_range": 99}},
        "pieces[1].moves[0].screen_range may not exceed 16",
        id="screen-range-max",
    ),
    # ---------------------------------------------------------------- condition
    pytest.param(
        {"pieces.1.moves.0.condition": 1}, "pieces[1].moves[0].condition must be an object",
        id="condition-type",
    ),
    pytest.param(
        {"pieces.1.moves.0.condition": {"from_rank": -1}},
        "pieces[1].moves[0].condition.from_rank must be a non-negative integer",
        id="condition-from-rank",
    ),
    pytest.param(
        {"pieces.1.moves.0.condition": {"first_move": 1}},
        "pieces[1].moves[0].condition.first_move must be a boolean",
        id="condition-first-move",
    ),
    # ---------------------------------------------------------------- setup
    pytest.param({"setup.mirror": "yes"}, "setup.mirror must be a boolean", id="setup-mirror"),
    pytest.param(
        {"setup.orientation": "spin"},
        f"setup.orientation must be one of {ORIENTATIONS}, got 'spin'",
        id="setup-orientation",
    ),
    pytest.param(
        {"setup.placements.2": "K"}, "setup.placements[2] must be an object",
        id="placement-type",
    ),
    pytest.param(
        {"setup.placements.1.player": 2}, "setup.placements[1].player must be 0 or 1, got 2",
        id="placement-player",
    ),
    pytest.param(
        {"setup.placements.1.player": 1},
        "setup.placements[1].player must be 0 when setup.mirror is true "
        "(the second player's army is derived by reflection)",
        id="placement-player-mirrored",
    ),
    pytest.param(
        {"setup.placements.1.squares": []},
        "setup.placements[1].squares must be a non-empty array of algebraic squares",
        id="placement-squares",
    ),
    pytest.param(
        {"pieces.1.facing": {"axes": 4}, "setup.placements.1.facing": "up"},
        f"setup.placements[1].facing must be one of {', '.join(FACING_NAMES)}",
        id="placement-facing",
    ),
    # ---------------------------------------------------------------- rules
    pytest.param({"rules": []}, "rules must be an object", id="rules-type"),
    pytest.param(
        {"rules.checkmate": "draw"}, "rules.checkmate must be 'loss' or 'win', got 'draw'",
        id="checkmate",
    ),
    pytest.param(
        {"rules.royal_facing": "sometimes"},
        "rules.royal_facing must be 'allowed' or 'forbidden', got 'sometimes'",
        id="royal-facing",
    ),
    pytest.param(
        {"rules.bare_royal": "maybe"},
        f"rules.bare_royal must be one of {ENDING_RESULTS}, got 'maybe'",
        id="bare-royal",
    ),
    pytest.param(
        {"rules.check_limit": 21}, "rules.check_limit must be an integer between 1 and 20",
        id="check-limit",
    ),
    pytest.param(
        {"rules.promotion": []}, "rules.promotion may not be an empty array",
        id="promotion-empty-list",
    ),
    # ---------------------------------------------------------------- explode_on_capture
    pytest.param(
        {"rules.explode_on_capture": True}, "rules.explode_on_capture must be an object",
        id="explosion-type",
    ),
    pytest.param(
        {"rules.explode_on_capture": {"radius": 4}},
        "rules.explode_on_capture.radius must be an integer between 0 and 3",
        id="explosion-radius",
    ),
    pytest.param(
        {"rules.explode_on_capture": {"by": "R"}},
        "rules.explode_on_capture.by must be an array of piece ids",
        id="explosion-by",
    ),
    # ---------------------------------------------------------------- drops
    # An empty list here; a string is in `UNHASHABLE` below.
    pytest.param({"rules.drops": []}, "rules.drops must be an object", id="drops-type"),
    pytest.param(
        {"rules.drops": {"from_captures": 1}}, "rules.drops.from_captures must be a boolean",
        id="drops-from-captures",
    ),
    pytest.param(
        {"rules.drops": {"placement": 1}}, "rules.drops.placement must be a boolean",
        id="drops-placement",
    ),
    pytest.param(
        {"rules.drops": {"either_face": 1}}, "rules.drops.either_face must be a boolean",
        id="drops-either-face",
    ),
    pytest.param(
        {"rules.drops": {"no_doubled": "R"}}, "rules.drops.no_doubled must be an array of piece ids",
        id="drops-no-doubled",
    ),
    pytest.param(
        {"rules.drops": {"region": ["a1"]}}, "rules.drops.region must be an object keyed by piece id",
        id="drops-region-type",
    ),
    pytest.param(
        {"rules.drops": {"region": {"Z": ["a1"]}}},
        "rules.drops.region key 'Z' is not a declared piece",
        id="drops-region-key",
    ),
    pytest.param(
        {"rules.drops": {"stock": {}}},
        "rules.drops.stock must be a non-empty object keyed by piece id, whose "
        f"values are counts of 1 to {MAX_STOCK} or {STOCK_UNLIMITED!r}",
        id="drops-stock-empty",
    ),
    # ---------------------------------------------------------------- reach
    pytest.param({"rules.reach": ["c3"]}, "rules.reach must be an object", id="reach-type"),
    pytest.param(
        {"rules.reach": {"squares": ["c3"], "mirror": "flip"}},
        'rules.reach.mirror must be a boolean or "rotate"',
        id="reach-mirror",
    ),
    pytest.param(
        {"rules.reach": {"squares": ["c3"], "safe": 1}}, "rules.reach.safe must be a boolean",
        id="reach-safe",
    ),
    pytest.param(
        {"rules.reach": {"squares": ["c3"], "result": "glory"}},
        f"rules.reach.result must be one of {ENDING_RESULTS}, got 'glory'",
        id="reach-result",
    ),
    pytest.param(
        {"rules.reach": {"squares": []}},
        "rules.reach.squares must be a non-empty array of algebraic squares",
        id="reach-squares",
    ),
    # ---------------------------------------------------------------- extinction
    pytest.param({"rules.extinction": "R"}, "rules.extinction must be an object", id="extinction-type"),
    pytest.param(
        {"rules.extinction": {"pieces": []}},
        "rules.extinction.pieces must be 'any' or a non-empty array of piece ids",
        id="extinction-pieces",
    ),
    pytest.param(
        {"rules.extinction": {"pieces": ["R"], "min_count": 0}},
        "rules.extinction.min_count must be an integer between 1 and 64",
        id="extinction-min-count",
    ),
    pytest.param(
        {"rules.extinction": {"pieces": ["R"], "result": "glory"}},
        f"rules.extinction.result must be one of {ENDING_RESULTS}, got 'glory'",
        id="extinction-result",
    ),
    pytest.param(
        {"rules.extinction": {"pieces": ["R", "K"], "per_type": 1}},
        'rules.extinction.per_type must be true, false or "own"',
        id="extinction-per-type",
    ),
    # ---------------------------------------------------------------- wrap, obstacles
    pytest.param({"rules.wrap": True}, "rules.wrap must be an object", id="wrap-type"),
    pytest.param({"rules.obstacles": ["c3"]}, "rules.obstacles must be an object", id="obstacles-type"),
    # ---------------------------------------------------------------- npcs
    pytest.param(
        {"rules.npcs": ["R"]}, "rules.npcs[0] must be an object with 'piece' and 'route'",
        id="npc-type",
    ),
    pytest.param(
        {"rules.npcs": [{"piece": "R", "route": ["b3", "c3"], "immune": "K"}]},
        "rules.npcs[0].immune must be an array of piece ids",
        id="npc-immune-type",
    ),
    # ---------------------------------------------------------------- turn
    pytest.param({"rules.turn": 2}, "rules.turn must be an object", id="turn-type"),
    pytest.param(
        {"rules.turn": {"moves": 9}}, "rules.turn.moves must be an integer between 1 and 4",
        id="turn-moves",
    ),
    pytest.param(
        {"rules.turn": {"moves": 2, "ramp": []}},
        "rules.turn.ramp must be a non-empty array of turn lengths",
        id="turn-ramp-empty",
    ),
    pytest.param(
        {"rules.turn": {"moves": 2, "ramp": [1] * 9}},
        "rules.turn.ramp may describe at most eight opening turns",
        id="turn-ramp-long",
    ),
    pytest.param(
        {"rules.turn": {"moves": 2, "ramp": [5]}},
        "rules.turn.ramp entry 5 must be an integer between 1 and 4",
        id="turn-ramp-entry",
    ),
    # ---------------------------------------------------------------- territory, scoring
    pytest.param({"rules.territory": 1}, "rules.territory must be an object", id="territory-type"),
    pytest.param(
        {"rules.territory": {"mode": "paint"}},
        "rules.territory.mode must be one of ('permanent', 'repaint'), got 'paint'",
        id="territory-mode",
    ),
    pytest.param(
        {"rules.territory": {"mode": "repaint", "win_at": -1}},
        "rules.territory.win_at must be a non-negative integer",
        id="territory-win-at",
    ),
    pytest.param(
        {"rules.territory": {"mode": "repaint", "win_at": 26}},
        "rules.territory.win_at is 26, which is more squares than the board has (25); "
        "nobody could reach it",
        id="territory-win-at-past-board",
    ),
    pytest.param({"rules.scoring": 1}, "rules.scoring must be an object", id="scoring-type"),
    pytest.param(
        {"rules.scoring": {"count": "material", "region": ["c3"]}},
        "rules.scoring.region only means something when count is 'region', not 'material'",
        id="scoring-region",
    ),
    pytest.param(
        {"rules.scoring": {"count": "territory"}},
        'rules.scoring.count "territory" needs rules.territory, or there is nothing '
        "coloured to compare",
        id="scoring-territory",
    ),
    # ---------------------------------------------------------------- strike, contacts
    pytest.param({"rules.strike": 1}, "rules.strike must be an object", id="strike-type"),
    pytest.param({"rules.contacts": 1}, "rules.contacts must be an object", id="contacts-type"),
    pytest.param(
        {"rules.contacts": {"immune": "K", "links": [{"mover": "R", "onto": "R", "onto_to": "K"}]}},
        "rules.contacts.immune must be an array of piece ids",
        id="contacts-immune-type",
    ),
    pytest.param(
        {"rules.contacts": {}}, "rules.contacts needs links: a table with no link is no rule",
        id="contacts-no-links",
    ),
    pytest.param(
        {"rules.contacts": {"links": [1]}},
        "rules.contacts.links[0] must be an object with 'mover' and 'onto'",
        id="contacts-link-type",
    ),
    pytest.param(
        {"rules.contacts": {"links": [{"mover": "R", "onto": "K", "onto_to": "K"}]}},
        "rules.contacts.links[0].onto_to points 'K' at itself, which changes nothing",
        id="contacts-onto-to-self",
    ),
    # ---------------------------------------------------------------- promotion
    pytest.param({"rules.promotion": 1}, "rules.promotion must be an object", id="promotion-type"),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "rank": 4, "ranks": [4]}},
        "rules.promotion has both 'rank' and 'ranks'; use one or the other",
        id="promotion-rank-and-ranks",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "rank": -1}},
        "rules.promotion.rank must be 'last' or a non-negative integer",
        id="promotion-rank",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "ranks": []}},
        "rules.promotion.ranks must be a non-empty array of ranks",
        id="promotion-ranks",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "mandatory": 1}},
        "rules.promotion.mandatory must be a boolean",
        id="promotion-mandatory",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "only_lost": 1}},
        "rules.promotion.only_lost must be a boolean",
        id="promotion-only-lost",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "to_by_player": {"0": ["K"], "1": ["K"]}}},
        "rules.promotion has both 'to' and 'to_by_player'; use one or the other",
        id="promotion-to-and-by-player",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to_by_player": ["K"]}},
        "rules.promotion.to_by_player must be an object keyed by '0' and '1'",
        id="promotion-by-player-type",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to_by_player": {"0": ["K"], "1": ["K"], "2": ["K"]}}},
        "rules.promotion.to_by_player has unknown players: 2",
        id="promotion-by-player-unknown",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to_by_player": {"0": ["K"]}}},
        "rules.promotion.to_by_player is missing player '1'",
        id="promotion-by-player-missing",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": []}},
        "rules.promotion.to must be a non-empty array of piece ids",
        id="promotion-targets",
    ),
    # ---------------------------------------------------------------- castling
    pytest.param({"rules.castling": 1}, "rules.castling must be an object", id="castling-type"),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R", "king_steps": 2,
                            "king_file": {"kingside": "d", "queenside": "b"}}},
        "rules.castling has both 'king_steps' and 'king_file'; a step count and a "
        "destination file are two ways of saying the same thing",
        id="castling-steps-and-file",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R", "king_steps": 5}},
        "rules.castling.king_steps must be an integer between 1 and 4",
        id="castling-steps",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R", "king_file": "d"}},
        "rules.castling.king_file must be an object with 'kingside' and 'queenside'",
        id="castling-file-type",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R",
                            "king_file": {"kingside": "d", "queenside": "b", "middle": "c"}}},
        "rules.castling.king_file has unknown sides: middle",
        id="castling-file-unknown-side",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R", "king_file": {"kingside": "d"}}},
        "rules.castling.king_file is missing 'queenside'",
        id="castling-file-missing-side",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R",
                            "king_file": {"kingside": "z", "queenside": "b"}}},
        "rules.castling.king_file.kingside 'z' is not a file on this board",
        id="castling-file-off-board",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R",
                            "rook_file": {"kingside": "d", "queenside": "b"}}},
        "rules.castling.rook_file requires king_file",
        id="castling-rook-file-alone",
    ),
]


@pytest.mark.parametrize(("changes", "refusal"), SHAPE)
def test_a_malformed_value_is_refused_in_words(changes: dict[str, Any], refusal: str) -> None:
    errors = validate_spec(_patched(BASE, changes))
    assert refusal in errors, errors


# Refusals about how rules sit together, and the spec's two free-text sections.
COHERENCE = [
    # ---------------------------------------------------------------- appearance
    pytest.param({"appearance": 1}, "appearance must be an object", id="appearance-type"),
    pytest.param(
        {"appearance": {"icons": 1}},
        "appearance.icons must be an object mapping piece id to icon name",
        id="icons-type",
    ),
    pytest.param(
        {"appearance": {"icons": {"Z": "rook"}}}, "appearance.icons names unknown piece 'Z'",
        id="icons-unknown-piece",
    ),
    pytest.param(
        {"appearance": {"icons": {"R": ["rook"]}}},
        "appearance.icons['R'] as a list must hold exactly two entries, one per player",
        id="icons-pair-length",
    ),
    pytest.param(
        {"appearance": {"icons": {"R": " "}}},
        "appearance.icons['R'] must be a non-empty string, or a pair of them for a piece "
        "drawn differently for each player",
        id="icons-blank",
    ),
    # ---------------------------------------------------------------- provenance
    pytest.param({"provenance": 1}, "provenance must be an object", id="provenance-type"),
    pytest.param(
        {"provenance": {"year": 1.5}},
        "provenance.year must be a year or a string such as '15th century'",
        id="provenance-year",
    ),
    pytest.param(
        {"provenance": {"parents": [1]}}, "provenance.parents must be a list of ruleset hashes",
        id="provenance-parents",
    ),
    # ---------------------------------------------------------------- pieces together
    pytest.param(
        {"pieces.1.promoted_from": "Z"},
        "piece 'R' promoted_from 'Z', which is not a declared piece",
        id="promoted-from-undeclared",
    ),
    pytest.param(
        {"rules.explode_on_capture": {"of": ["Z"]}},
        "rules.explode_on_capture.of entry 'Z' is not a declared piece",
        id="explosion-of-undeclared",
    ),
    pytest.param(
        {"rules.npcs": [{"piece": "R", "route": "b3"}]},
        "rules.npcs[0].route must be a non-empty array of algebraic squares",
        id="npc-route-type",
    ),
    # ---------------------------------------------------------------- strike
    pytest.param(
        {"rules.strike": {"position": "fly"}},
        "rules.strike.position must be one of ('stay', 'advance', 'swap', 'choose', 'home'), "
        "got 'fly'",
        id="strike-position",
    ),
    pytest.param(
        {"rules.strike": {"switches_sides": ["Z"]}},
        "rules.strike.switches_sides entry 'Z' is not a declared piece",
        id="strike-switches-undeclared",
    ),
    pytest.param(
        {"rules.strike": {"switches_sides": ["R", "R"]}},
        "rules.strike.switches_sides names a piece twice",
        id="strike-switches-twice",
    ),
    pytest.param(
        {"rules.strike": {"immune": "K"}}, "rules.strike.immune must be an array of piece ids",
        id="strike-immune-type",
    ),
    pytest.param(
        {"rules.strike": {"immune": ["Z"]}}, "rules.strike.immune entry 'Z' is not a declared piece",
        id="strike-immune-undeclared",
    ),
    pytest.param(
        {"rules.strike": {"effect": "demote", "chain": []}},
        "rules.strike.chain must be a non-empty array of from/to links",
        id="strike-chain-empty",
    ),
    pytest.param(
        {"rules.strike": {"effect": "demote", "chain": [1]}},
        "rules.strike.chain[0] must be an object with 'from' and 'to'",
        id="strike-link-type",
    ),
    pytest.param(
        {"rules.strike": {"effect": "demote", "chain": [{"from": 1, "to": "R"}]}},
        "rules.strike.chain[0].from 1 is not a declared piece",
        id="strike-link-from",
    ),
    # ---------------------------------------------------------------- contacts
    pytest.param(
        {"rules.contacts": {"immune": ["Z"], "links": [{"mover": "R", "onto": "R", "onto_to": "K"}]}},
        "rules.contacts.immune entry 'Z' is not a declared piece",
        id="contacts-immune-undeclared",
    ),
    pytest.param(
        {"rules.contacts": {"links": [{"mover": 1, "onto": "R", "onto_to": "K"}]}},
        "rules.contacts.links[0].mover 1 is not a declared piece",
        id="contacts-link-mover",
    ),
    # ---------------------------------------------------------------- scoring
    pytest.param(
        {"rules.scoring": {"count": "goals"}},
        "rules.scoring.count must be one of ('material', 'pieces', 'region', 'territory', "
        "'points'), got 'goals'",
        id="scoring-count",
    ),
]


@pytest.mark.parametrize(("changes", "refusal"), COHERENCE)
def test_a_rule_that_does_not_fit_is_refused_in_words(
    changes: dict[str, Any], refusal: str
) -> None:
    errors = validate_spec(_patched(BASE, changes))
    assert refusal in errors, errors


ONE_PORTAL = "piece 'O' (a portal, rules.portals)"
PORTAL_TYPE = "piece 'O' is a portal (rules.portals)"
LEAVE = {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"}

# Schema 19: the shape of a pair, and what a portal type may not be.
PORTAL = [
    pytest.param(
        {"rules.portals": []}, "rules.portals must be a non-empty array of portal groups",
        id="portals-empty",
    ),
    pytest.param(
        {"rules.portals.1": "b4"},
        "rules.portals[1] must be an object with 'piece' and either 'orange' and 'blue' or 'ends'",
        id="pair-type",
    ),
    pytest.param(
        {"rules.portals.0.piece": "Z"}, "rules.portals[0].piece 'Z' is not a declared piece",
        id="pair-piece",
    ),
    pytest.param(
        {"rules.portals.0.blue": "b3"},
        "rules.portals[0] puts both ends on b3; a pair needs two squares",
        id="pair-one-square",
    ),
    pytest.param(
        {"rules.portals.0.moved_by": [0]},
        "rules.portals[0].moved_by must be an object keyed by 'orange' and 'blue'",
        id="moved-by-type",
    ),
    pytest.param(
        {"rules.portals.0.moved_by": {"green": [0]}},
        "rules.portals[0].moved_by has unknown key 'green'; it names the ends 'orange' and "
        "'blue'",
        id="moved-by-key",
    ),
    pytest.param(
        {"rules.portals.0.moved_by": {"orange": [2]}},
        "rules.portals[0].moved_by.orange must be a list of distinct players, drawn from 0 "
        "and 1 ([] for nobody)",
        id="moved-by-players",
    ),
    pytest.param(
        {"rules.portals.0.passage": "open"},
        "rules.portals[0].passage must be an object keyed by player, '0' and '1'",
        id="passage-type",
    ),
    pytest.param(
        {"rules.portals.0.passage": {"2": "closed"}},
        "rules.portals[0].passage has unknown key '2'; it is keyed by player, '0' and '1'",
        id="passage-key",
    ),
    pytest.param(
        {"pieces.1.region": ["a1", "b1"]},
        "piece 'R' has a region, which rules.portals refuses in this schema: an exit could "
        "put the piece outside it",
        id="region-beside-portals",
    ),
    pytest.param({"pieces.2.royal": True}, f"{ONE_PORTAL} may not be royal", id="portal-royal"),
    pytest.param(
        {"pieces.2.captures": []}, f"{ONE_PORTAL} may not declare captures", id="portal-captures"
    ),
    pytest.param(
        {"pieces.2.moves": [{**LEAVE, "kind": "sweep"}]}, f"{ONE_PORTAL} moves[0] may not sweep",
        id="portal-sweeps",
    ),
    pytest.param(
        {"pieces.2.moves": [{**LEAVE, "condition": {"from_rank": 0}}]},
        f"{ONE_PORTAL} moves[0] may not have a from_rank or region condition: those are read "
        "from a player's side, and a portal belongs to neither",
        id="portal-condition",
    ),
    pytest.param(
        {"rules.drops": {"no_mate": ["O"]}},
        f"{PORTAL_TYPE} and may not be named by rules.drops: it is never held",
        id="portal-dropped",
    ),
    pytest.param(
        {"rules.extinction": {"pieces": ["O"]}},
        f"{PORTAL_TYPE} and may not be counted by rules.extinction",
        id="portal-extinction",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "O"}}, f"{PORTAL_TYPE} and may not castle",
        id="portal-castles",
    ),
    pytest.param(
        {"pieces.1.captures": ["K", "O"]},
        "piece 'R' lists 'O' in captures, but a portal is never captured",
        id="portal-captured",
    ),
    pytest.param(
        {"pieces.1.promoted_from": "O"}, "piece 'R' is promoted_from 'O', a portal",
        id="promoted-from-portal",
    ),
]


@pytest.mark.parametrize(("changes", "refusal"), PORTAL)
def test_a_portal_rule_that_does_not_fit_is_refused_in_words(
    changes: dict[str, Any], refusal: str
) -> None:
    errors = validate_spec(_patched(PORTALS, changes))
    assert refusal in errors, errors


# Malformed values the validator once crashed on instead of refusing (found by F8,
# fixed by Q825): a string where an object belongs, and a list or object where a piece id
# belongs, which a set membership test raised on. Each is refused by its shape check.
UNHASHABLE = [
    pytest.param({"rules.drops": "stock"}, "rules.drops must be an object", id="drops-a-string"),
    pytest.param(
        {"pieces.1.captures": [{"id": "K"}]},
        "pieces[1].captures entry {'id': 'K'} must be a piece id",
        id="captures-entry-an-object",
    ),
    pytest.param(
        {"pieces.1.id": ["R"]}, "pieces[1].id must be a string of 1-2 characters",
        id="piece-id-a-list",
    ),
    pytest.param(
        {"pieces.1.promoted_from": ["K"]}, "pieces[1].promoted_from must be a piece id",
        id="promoted-from-a-list",
    ),
    pytest.param(
        {"setup.placements.1.piece": ["R"]},
        "setup.placements[1].piece ['R'] is not a declared piece id",
        id="placement-piece-a-list",
    ),
]


@pytest.mark.parametrize(("changes", "refusal"), UNHASHABLE)
def test_a_malformed_value_is_refused_not_raised_on(changes: dict[str, Any], refusal: str) -> None:
    errors = validate_spec(_patched(BASE, changes))
    assert refusal in errors, errors


PAWN = {
    "id": "P",
    "moves": [
        {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1], [-1, 1]], "mode": "capture"},
    ],
}
WITH_PAWN = {"pieces.2": PAWN, "setup.placements.2": {"piece": "P", "squares": ["b2"]}}

# Refusals whose branch is a route through a helper rather than a value's shape.
ROUTES = [
    pytest.param(
        {"schema_version": 1, "setup.placements.0.squares": ["c1", "e1"]},
        "more than one royal piece under 'checkmate' requires \"schema_version\": 2",
        id="two-royals-placed-under-schema-1",
    ),
    pytest.param(
        {"schema_version": 1,
         "rules.contacts": {"links": [{"mover": "R", "onto": "R", "onto_to": "K"}]}},
        "more than one royal piece under 'checkmate' requires \"schema_version\": 2",
        id="contact-makes-a-royal-under-schema-1",
    ),
    pytest.param(
        {"rules.promotion": [{"piece": "R", "to": ["K"]}, {"piece": "R", "to": ["K"]}]},
        "rules.promotion has two rules for 'R'; a piece promotes one way",
        id="promotion-two-rules-for-a-piece",
    ),
    pytest.param(
        {"rules.en_passant": True,
         "rules.promotion": [{"piece": "R", "to": ["K"]}, {"piece": "K", "to": ["R"]}]},
        "rules.en_passant needs a single promoting piece to be defined against, but "
        "rules.promotion names 2",
        id="en-passant-two-promoters",
    ),
    pytest.param(
        {"rules.promote_on_entry": True,
         "rules.promotion": {"piece": "R", "to": ["K"], "squares": ["c5"]},
         "rules.drops": {"region": {"R": ["a3"]}}},
        "rules.promote_on_entry can never fire: no piece with a promotion rule can enter "
        "play inside its zone -- R may not be dropped on c5",
        id="entry-zone-named-by-square",
    ),
    pytest.param(
        {"setup.placements.1.squares": ["1a"]},
        "setup.placements[1].squares entry '1a' is not a square on this board",
        id="square-file-not-a-letter",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R",
                            "king_file": {"kingside": "dd", "queenside": "b"}}},
        "rules.castling.king_file.kingside 'dd' is not a file on this board",
        id="castling-file-two-letters",
    ),
]


@pytest.mark.parametrize(("changes", "refusal"), ROUTES)
def test_a_refusal_reached_through_a_route_is_said_in_words(
    changes: dict[str, Any], refusal: str
) -> None:
    errors = validate_spec(_patched(BASE, changes))
    assert refusal in errors, errors


def test_portal_ends_count_against_the_piece_type_limit() -> None:
    declared = MAX_PIECE_TYPES - 1  # one under the limit, and one pair adds two types
    extra = [{"id": f"X{i}", "moves": [STEP]} for i in range(declared - len(PORTALS["pieces"]))]
    spec = _patched(PORTALS, {"pieces": PORTALS["pieces"] + extra})
    assert (
        f"at most {MAX_PIECE_TYPES} piece types, and rules.portals adds one per end: "
        f"{declared} declared and 2 portal ends"
    ) in validate_spec(spec)


# A spec already refused for its shape reaches the checks that read rules together, and
# each of them skips what it cannot read rather than crashing or adding a refusal that is
# only a consequence of the first ("an inert-rule refusal has to be a fact, not a
# consequence of a typo somewhere else", `_capture_is_possible`). Each row's errors are
# exactly the shape refusals listed.
CONSEQUENCES = [
    pytest.param(
        BASE, {"pieces.2": "x", "rules.promotion": {"piece": "R", "to": ["K"], "only_lost": True}},
        ["pieces[2] must be an object"],
        id="only-lost-skips-a-piece-that-is-not-an-object",
    ),
    pytest.param(
        BASE, {"schema_version": 1, "pieces.1.moves.0": "x"},
        ["pieces[1].moves[0] must be an object"],
        id="schema-1-gate-skips-a-move-that-is-not-an-object",
    ),
    pytest.param(
        BASE, {"schema_version": 1, "pieces.2": "x", "rules.drops": {}},
        ["pieces[2] must be an object",
         'rules uses drops, added in schema 2; declare "schema_version": 2 to use it'],
        id="schema-1-gate-skips-a-piece-that-is-not-an-object",
    ),
    pytest.param(
        BASE, {"setup.placements": "x", "rules.castling": {"king": "K", "rook": "R"},
               "rules.obstacles": {"squares": ["c3"]}},
        ["setup.placements must be a non-empty array"],
        id="castling-beside-walls-skips-a-missing-army",
    ),
    pytest.param(
        BASE, {"rules.castling": {"king": "K", "rook": "R",
                                  "king_file": {"kingside": "dd", "queenside": "b"}},
               "rules.obstacles": {"squares": ["c3"]}},
        ["rules.castling.king_file.kingside 'dd' is not a file on this board"],
        id="castling-beside-walls-skips-an-unreadable-file",
    ),
    pytest.param(
        BASE, {"rules.drops": {"stock": "x", "no_mate": ["R"]}},
        ["rules.drops.stock must be a non-empty object keyed by piece id, whose values are "
         f"counts of 1 to {MAX_STOCK} or {STOCK_UNLIMITED!r}"],
        id="drop-rule-skips-a-malformed-stock",
    ),
    pytest.param(
        BASE, {"rules.drops": {"from_captures": 1, "no_mate": ["R"]}},
        ["rules.drops.from_captures must be a boolean"],
        id="drop-rule-skips-a-malformed-from-captures",
    ),
    pytest.param(
        BASE, {"rules.contacts": {"position": "stay"}},
        ["rules.contacts needs links: a table with no link is no rule"],
        id="contacts-without-links-is-refused-once",
    ),
    pytest.param(
        PORTALS, {"rules.promotion": ["x"]}, ["rules.promotion[0] must be an object"],
        id="portal-check-skips-a-promotion-that-is-not-an-object",
    ),
    # A board that cannot be read places nobody, so "starts with none" would repeat the
    # board's refusal (F8's noted echo, Q825); the squares it cannot place are refused.
    *(
        pytest.param(
            BASE, {**board, "rules.promotion": {"piece": "R", "to": ["K"], "only_lost": True}},
            [refusal,
             "setup.placements[0].squares entry 'c1' is not a square on this board",
             "setup.placements[1].squares entry 'a1' is not a square on this board"],
            id=f"only-lost-skips-{name}",
        )
        for name, board, refusal in (
            ("a-board-that-is-not-an-object", {"board": "x"},
             "board must be an object with integer width and height"),
            ("a-width-that-is-not-an-integer", {"board.width": "5"},
             "board.width must be an integer"),
            ("a-width-off-the-range", {"board.width": 0},
             f"board.width must be between 2 and {MAX_BOARD_DIM}, got 0"),
        )
    ),
]


@pytest.mark.parametrize(("base", "changes", "refusals"), CONSEQUENCES)
def test_a_malformed_spec_draws_only_its_own_refusals(
    base: dict[str, Any], changes: dict[str, Any], refusals: list[str]
) -> None:
    assert validate_spec(_patched(base, changes)) == refusals


def test_an_appearance_with_no_icons_is_not_refused() -> None:
    assert validate_spec(_patched(BASE, {"appearance": {}})) == []


# Two inert-rule refusals, each first drawn by a well-formed spec and then withheld when a
# malformed piece makes the answer unknowable: the malformation is refused, and the rule
# is not also called inert because of it.
NOTHING_TAKES = {"rules.must_capture": True, "pieces.0.captures": [], "pieces.1.captures": []}
NEVER_INERT = (
    "rules.must_capture can never bind: no piece has a capture-mode move that rules the "
    "capture matrix permits, so nothing can ever be taken, so there is never a capture to "
    "prefer"
)
ONE_STEP = {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}
NO_DOUBLE_STEP = {
    "pieces.2": {"id": "P", "moves": [ONE_STEP]},
    "setup.placements.2": {"piece": "P", "squares": ["b2"]},
    "rules.en_passant": True,
    "rules.promotion": {"piece": "P", "to": ["R"]},
}
EN_PASSANT_INERT = (
    "rules.en_passant can generate no capture here: it answers a two-square move, and "
    "'P', the promoting piece it is defined against, has none -- a leap passes over "
    "nothing, a hop passes over its own screen, and a slide needs a range of at least 2 "
    "(or 0 for unlimited), a min_range of at most 2, and a mode that moves. It is charged "
    "1.0 all the same."
)


@pytest.mark.parametrize(
    ("changes", "refusals"),
    [
        pytest.param({}, [NEVER_INERT], id="well-formed"),
        pytest.param(
            {"pieces.0.moves": "x"}, ["pieces[0].moves must be a non-empty array"],
            id="moves-not-a-list",
        ),
        pytest.param(
            {"pieces.0.moves.0": "x"}, ["pieces[0].moves[0] must be an object"],
            id="a-move-not-an-object",
        ),
    ],
)
def test_must_capture_is_called_inert_only_when_that_is_a_fact(
    changes: dict[str, Any], refusals: list[str]
) -> None:
    assert validate_spec(_patched(BASE, {**NOTHING_TAKES, **changes})) == refusals


@pytest.mark.parametrize(
    ("changes", "refusals"),
    [
        pytest.param({}, [EN_PASSANT_INERT], id="well-formed"),
        pytest.param(
            {"pieces.2.moves": "x"}, ["pieces[2].moves must be a non-empty array"],
            id="moves-not-a-list",
        ),
        pytest.param(
            {"pieces.2.moves": [ONE_STEP, "x"]}, ["pieces[2].moves[1] must be an object"],
            id="a-move-not-an-object",
        ),
        pytest.param(
            {"pieces.2.moves": [{**ONE_STEP, "kind": "slide", "range": "2"}]},
            ["pieces[2].moves[0].range must be a non-negative integer"],
            id="a-range-not-a-number",
        ),
    ],
)
def test_en_passant_is_called_inert_only_when_that_is_a_fact(
    changes: dict[str, Any], refusals: list[str]
) -> None:
    assert validate_spec(_patched(BASE, {**NO_DOUBLE_STEP, **changes})) == refusals


# ------------------------------------------------------------------ the edges of each bound
#
# Q739 (mutation testing of `spec/`, `docs/reference/mutation/spec.md`): a bound that moved
# by one, or a loop that stopped at its first bad entry, failed nothing. Each ACCEPTED row
# is a spec on the permitted side of a bound and must validate clean; each EDGE_REFUSED row
# is the neighbouring value or the second of two faults, and must draw the exact words.

WARP = {"kind": "warp", "vectors": [[1, 0]]}  # a kind the validator refuses, so range defaults
HOP = {"kind": "hop", "vectors": [[1, 0]]}
PLAIN_PLACEMENTS = [
    {"piece": "K", "squares": ["c1"]},
    {"piece": "R", "squares": ["a1"]},
]
UNMIRRORED = {
    "setup": {
        "mirror": False,
        "placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "K", "squares": ["c5"], "player": 1},
            {"piece": "R", "squares": ["c2"], "player": 1},
        ],
    },
}
TWO_ROUTES = [
    {"piece": "R", "route": ["b3", "c3"], "immune": ["K"]},
    {"piece": "R", "route": ["d3", "d4"], "immune": ["K"]},
]
LINK = {"mover": "R", "onto": "K", "onto_to": "R"}

ACCEPTED = [
    pytest.param({"pieces.1.value": 0}, id="piece-value-zero"),
    pytest.param({"pieces.1.value": 100}, id="piece-value-hundred"),
    pytest.param(
        {"pieces.1.moves": [ROOK] * MAX_MOVE_DESCRIPTORS}, id="descriptors-at-the-limit"
    ),
    pytest.param({"pieces.1.moves.0.range": 0}, id="range-zero-is-unlimited"),
    pytest.param({"pieces.1.moves.0.range": 16}, id="range-at-the-limit"),
    pytest.param(
        {"pieces.1.moves.0": {**ROOK, "range": 0, "min_range": 2}},
        id="min-range-beyond-an-unlimited-range",
    ),
    pytest.param(
        {"pieces.1.moves.0": {**HOP, "screen_range": 16}}, id="screen-range-at-the-limit"
    ),
    pytest.param({"pieces.1.moves.0.vectors": [[8, -8]]}, id="vector-component-at-the-limit"),
    pytest.param({"pieces.1.moves.0.condition": {"from_rank": 0}}, id="from-rank-zero"),
    pytest.param(UNMIRRORED, id="one-square-two-players-unmirrored"),
    pytest.param({"rules.max_plies": 2}, id="clock-at-its-low-bound"),
    pytest.param({"rules.check_limit": 1}, id="check-limit-one"),
    pytest.param({"rules.check_limit": 20}, id="check-limit-twenty"),
    pytest.param(
        {"rules.en_passant": True, "rules.promotion": [{"piece": "R", "to": ["K"]}]},
        id="en-passant-beside-one-promotion-in-a-list",
    ),
    pytest.param(
        {"rules.promotion": [{"piece": "R", "to": ["K"]}, {"piece": "K", "to": ["R"]}]},
        id="two-promotions-without-en-passant",
    ),
    pytest.param({"rules.explode_on_capture": {"radius": 0}}, id="radius-zero"),
    pytest.param({"rules.explode_on_capture": {"radius": 3}}, id="radius-three"),
    pytest.param({"rules.reach": {"squares": ["c3", "a1"]}}, id="reach-squares-on-the-board"),
    pytest.param({"rules.extinction": {"pieces": ["R"], "min_count": 1}}, id="min-count-one"),
    pytest.param({"rules.turn": {"moves": 1}}, id="turn-moves-one"),
    pytest.param({"rules.turn": {"moves": 4}}, id="turn-moves-four"),
    pytest.param({"rules.turn": {"moves": 2, "ramp": [1] * 8}}, id="ramp-of-eight"),
    pytest.param({"rules.turn": {"moves": 2, "ramp": [1, 4]}}, id="ramp-entries-at-the-bounds"),
    pytest.param({"rules.territory": {"mode": "repaint", "win_at": 0}}, id="win-at-zero"),
    pytest.param({"rules.territory": {"mode": "repaint", "win_at": 25}}, id="win-at-every-square"),
    pytest.param({"rules.promotion": {"piece": "R", "to": ["K"], "rank": 0}}, id="rank-zero"),
    pytest.param({"rules.promotion": {"piece": "R", "to": ["K"], "rank": 4}}, id="rank-top"),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "ranks": [0, 4]}}, id="ranks-on-the-board"
    ),
    pytest.param({"rules.castling": {"king": "K", "rook": "R", "king_steps": 1}}, id="steps-one"),
    pytest.param({"rules.castling": {"king": "K", "rook": "R", "king_steps": 4}}, id="steps-four"),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R",
                            "king_file": {"kingside": "a", "queenside": "e"}}},
        id="castling-files-at-the-board-edges",
    ),
]


@pytest.mark.parametrize("changes", ACCEPTED)
def test_a_value_on_the_permitted_side_of_a_bound_is_accepted(changes: dict[str, Any]) -> None:
    assert validate_spec(_patched(BASE, changes)) == []


def test_min_count_64_is_inside_the_bound() -> None:
    """64 passes the bound; the base starts with too few rooks, which is a refusal of its own.

    No board this base can grow to holds 64 of a kind per side, so the top of the bound is
    held by what is *not* said: the start refusal (Q937), and not *"between 1 and 64"*.
    """
    errors = validate_spec(_patched(BASE, {"rules.extinction": {"pieces": ["R"], "min_count": 64}}))
    assert errors and all("so the game would be over before the first move" in e for e in errors)
    assert not any("between 1 and 64" in e for e in errors)


EDGE_REFUSED = [
    pytest.param(
        {"pieces.1.value": -1}, "pieces[1].value must be between 0 and 100", id="value-below-zero"
    ),
    pytest.param(
        {"pieces.1.moves": [ROOK] * (MAX_MOVE_DESCRIPTORS + 1)},
        f"pieces[1].moves may hold at most {MAX_MOVE_DESCRIPTORS} descriptors",
        id="one-descriptor-too-many",
    ),
    pytest.param(
        {"pieces.1.moves.0.range": 17}, "pieces[1].moves[0].range may not exceed 16",
        id="range-past-the-limit",
    ),
    pytest.param(
        {"pieces.1.moves.0": {**WARP, "min_range": 2}},
        "pieces[1].moves[0].min_range (2) may not exceed range (1)",
        id="min-range-past-the-default-range",
    ),
    pytest.param(
        {"pieces.1.moves.0": {**ROOK, "range": 1, "min_range": 2}},
        "pieces[1].moves[0].min_range (2) may not exceed range (1)",
        id="min-range-past-a-range-of-one",
    ),
    pytest.param(
        {"pieces.1.moves.0": {**HOP, "screen_range": 17}},
        "pieces[1].moves[0].screen_range may not exceed 16",
        id="screen-range-past-the-limit",
    ),
    pytest.param(
        {"pieces.1.moves.0.vectors": [[1], [1, 2, 3]]},
        "pieces[1].moves[0].vectors entries must be [dx, dy] integer pairs, got [1, 2, 3]",
        id="vectors-second-bad-pair-is-still-said",
    ),
    pytest.param(
        {"pieces.1.moves.0.vectors": [[0, -9]]},
        "pieces[1].moves[0].vectors components must be within +/-8, got [0, -9]",
        id="vector-component-below-the-limit",
    ),
    pytest.param(
        {"pieces.1.moves.0.vectors": [[9, 0]]},
        "pieces[1].moves[0].vectors components must be within +/-8, got [9, 0]",
        id="vector-component-above-the-limit",
    ),
    pytest.param(
        {"pieces.1.moves.0.condition": {"from_rank": "1"}},
        "pieces[1].moves[0].condition.from_rank must be a non-negative integer",
        id="from-rank-a-string",
    ),
    pytest.param(
        {"setup.mirror": DROP, "setup.placements.1.player": 1},
        "setup.placements[1].player must be 0 when setup.mirror is true "
        "(the second player's army is derived by reflection)",
        id="mirror-is-on-unless-said-otherwise",
    ),
    pytest.param(
        {"setup.placements.1.squares": ["z9", "y9"]},
        "setup.placements[1].squares entry 'y9' is not a square on this board",
        id="second-bad-square-is-still-said",
    ),
    pytest.param(
        {"setup.mirror": False, "setup.placements.1.player": "x",
         "setup.placements.1.squares": ["c1"]},
        "setup.placements[1].squares 'c1' is already occupied by 'K'",
        id="a-player-that-is-not-a-number-sits-with-player-zero",
    ),
    pytest.param(
        {"rules.no_progress_draw": 1}, "rules.no_progress_draw must be between 2 and 1000, got 1",
        id="a-later-clock-is-checked-when-an-earlier-one-is-absent",
    ),
    pytest.param(
        {"rules.check_limit": 0}, "rules.check_limit must be an integer between 1 and 20",
        id="check-limit-zero",
    ),
    pytest.param(
        {"rules.check_limit": "3"}, "rules.check_limit must be an integer between 1 and 20",
        id="check-limit-a-string",
    ),
    pytest.param(
        {"rules.explode_on_capture": {"radius": -1}},
        "rules.explode_on_capture.radius must be an integer between 0 and 3",
        id="radius-below-zero",
    ),
    pytest.param(
        {"rules.drops": {"stock": {"K": STOCK_UNLIMITED, "R": 0}}},
        f"rules.drops.stock['R'] must be between 1 and {MAX_STOCK}, got 0",
        id="a-stock-after-an-unlimited-one-is-still-checked",
    ),
    pytest.param(
        {"rules.reach": {"squares": ["c3", "z9"]}},
        "rules.reach.squares entry 'z9' is not a square on this board",
        id="reach-square-off-the-board",
    ),
    pytest.param(
        {"rules.reach": {"squares": ["c3", 7]}},
        "rules.reach.squares entry 7 is not a square on this board",
        id="reach-square-not-a-string",
    ),
    pytest.param(
        {"rules.extinction": {"pieces": "R"}},
        "rules.extinction.pieces must be 'any' or a non-empty array of piece ids",
        id="extinction-pieces-a-string",
    ),
    pytest.param(
        {"rules.extinction": {"pieces": ["R"], "min_count": 65}},
        "rules.extinction.min_count must be an integer between 1 and 64",
        id="min-count-65",
    ),
    pytest.param(
        {"rules.extinction": {}}, "rules.extinction is missing required key 'pieces'",
        id="a-required-key-is-named-when-missing",
    ),
    pytest.param(
        {"rules.npcs": [TWO_ROUTES[0], {**TWO_ROUTES[1], "immune": []}]},
        "rules.npcs has two wild 'R' with different 'immune' lists; capture permission is a "
        "fact about a piece type, so one type has one immunity. Declare a second type if "
        "the two hazards differ.",
        id="two-wild-pieces-of-a-type-disagree-about-immunity",
    ),
    pytest.param(
        {"rules.contacts": {"links": [LINK, LINK]}},
        "rules.contacts.links has two links for 'R' onto 'K'; one meeting has one outcome",
        id="contacts-twice-the-same-meeting",
    ),
    pytest.param(
        {"rules.contacts": {"links": [{"mover": "R", "onto": "K", "onto_to": "Z"}]}},
        "rules.contacts.links[0].onto_to 'Z' is not a declared piece",
        id="contacts-onto-to-checked-when-mover-to-is-absent",
    ),
    pytest.param(
        {"rules.turn": {"moves": 0}}, "rules.turn.moves must be an integer between 1 and 4",
        id="turn-moves-zero",
    ),
    pytest.param(
        {"rules.turn": {"moves": True}}, "rules.turn.moves must be an integer between 1 and 4",
        id="turn-moves-a-boolean",
    ),
    pytest.param(
        {"rules.turn": {"moves": 2, "ramp": [1] * 9}},
        "rules.turn.ramp may describe at most eight opening turns",
        id="ramp-of-nine",
    ),
    pytest.param(
        {"rules.territory": {"mode": "repaint", "win_at": 26}},
        "rules.territory.win_at is 26, which is more squares than the board has (25); "
        "nobody could reach it",
        id="win-at-one-past-the-board",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "rank": "first"}},
        "rules.promotion.rank must be 'last' or a non-negative integer",
        id="rank-a-word",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "ranks": "top"}},
        "rules.promotion.ranks must be a non-empty array of ranks",
        id="ranks-a-word",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "ranks": [4, 5]}},
        "rules.promotion.ranks entry 5 is not a rank on this board",
        id="ranks-entry-one-past-the-top",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "ranks": [-1]}},
        "rules.promotion.ranks entry -1 is not a rank on this board",
        id="ranks-entry-below-zero",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "ranks": [True]}},
        "rules.promotion.ranks entry True is not a rank on this board",
        id="ranks-entry-a-boolean",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "ranks": ["1"]}},
        "rules.promotion.ranks entry '1' is not a rank on this board",
        id="ranks-entry-a-string",
    ),
    pytest.param(
        {"rules.promotion": {"piece": "R", "to": ["K"], "rank": 5}},
        "rules.promotion.rank 5 is not a rank on this board",
        id="rank-one-past-the-top",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R", "king_steps": 0}},
        "rules.castling.king_steps must be an integer between 1 and 4",
        id="steps-zero",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R", "king_file": {"queenside": "z"}}},
        "rules.castling.king_file.queenside 'z' is not a file on this board",
        id="castling-queenside-checked-when-kingside-is-missing",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R",
                            "rook_file": {"kingside": "z", "queenside": "a"}}},
        "rules.castling.rook_file.kingside 'z' is not a file on this board",
        id="castling-rook-file-checked-when-king-file-is-absent",
    ),
    pytest.param(
        {"board.height": 1, "setup": DROP, "rules.promotion": {"piece": "R", "to": ["K"],
                                                               "ranks": [1]}},
        "rules.promotion.ranks entry 1 is not a rank on this board",
        id="ranks-entry-one-past-a-one-rank-board",
    ),
    pytest.param(
        {"rules.castling": {"king": "K", "rook": "R",
                            "king_file": {"kingside": "f", "queenside": "a"}}},
        "rules.castling.king_file.kingside 'f' is not a file on this board",
        id="castling-file-one-past-the-board",
    ),
]


@pytest.mark.parametrize(("changes", "refusal"), EDGE_REFUSED)
def test_the_value_past_a_bound_is_refused_in_the_exact_words(
    changes: dict[str, Any], refusal: str
) -> None:
    errors = validate_spec(_patched(BASE, changes))
    assert refusal in errors, errors


def test_a_player_that_is_a_number_does_not_sit_with_player_zero() -> None:
    # Without a mirror both armies share one board (Q924): player 1's rook on c1 beside
    # player 0's king is two pieces on a square, and is refused by naming it.
    clash = _patched(BASE, UNMIRRORED)
    clash["setup"]["placements"][2]["squares"] = ["c1"]
    errors = validate_spec(clash)
    assert "setup.placements[2].squares 'c1' is already occupied by 'K'" in errors, errors


def test_a_contact_between_unnamed_pieces_is_not_called_a_repeat() -> None:
    link = {"mover": 1, "onto": "K", "onto_to": "R"}
    errors = validate_spec(_patched(BASE, {"rules.contacts": {"links": [link, link]}}))
    assert "rules.contacts.links[0].mover 1 is not a declared piece" in errors
    assert not [e for e in errors if "has two links" in e], errors


@pytest.mark.parametrize(
    ("name", "width", "index"),
    [("a", 5, 0), ("A", 5, 0), ("e", 5, 4), ("f", 5, None), ("1", 5, None), ("ab", 5, None)],
)
def test_a_file_letter_is_read_against_the_board_width(
    name: str, width: int, index: int | None
) -> None:
    assert parse_file(name, width) == index


def test_npc_routes_skips_what_is_not_a_route_and_expands_a_mirror() -> None:
    rules = {"npcs": ["R", {"piece": "R", "route": "b2"},
                      {"piece": "R", "route": ["b2", "c2"], "mirror": True}]}
    assert npc_routes(rules, 5, 5) == [
        ("rules.npcs[2]", (6, 7)),
        ("rules.npcs[2] (mirrored)", (16, 17)),
    ]


def test_a_board_with_no_ranks_is_left_to_the_board_check() -> None:
    # A height the board check refused reaches the promotion checks as 0: it is that
    # check's to report, so a rank or a band is not also called off a board that has none.
    spec = _patched(BASE, {"board.height": 1, "rules.promotion": {"piece": "R", "to": ["K"],
                                                                 "rank": 0, "ranks": [0]}})
    errors = validate_spec(spec)
    assert "board.height must be between 2 and 16, got 1" in errors
    assert not [e for e in errors if "is not a rank on this board" in e], errors


# ----------------------------- F-spec-last (Q946): the last 11 missing `spec/` tests, `spec.md`


@pytest.mark.parametrize("value", [True, "3", 2.5])
def test_extinction_min_count_that_is_not_an_integer_is_refused_not_raised_on(value: Any) -> None:
    errors = validate_spec(_patched(BASE, {"rules.extinction": {"pieces": ["R"], "min_count": value}}))
    assert "rules.extinction.min_count must be an integer between 1 and 64" in errors, errors


@pytest.mark.parametrize("value", [True, "3", 2.5])
def test_territory_win_at_that_is_not_an_integer_is_refused_not_raised_on(value: Any) -> None:
    errors = validate_spec(_patched(BASE, {"rules.territory": {"mode": "repaint", "win_at": value}}))
    assert "rules.territory.win_at must be a non-negative integer" in errors, errors


def test_a_drop_restriction_after_an_absent_one_is_still_checked() -> None:
    # `no_doubled` is absent, so the loop must go on to `no_mate`, not stop there.
    errors = validate_spec(_patched(BASE, {"rules.drops": {"no_mate": ["Z"]}}))
    assert "rules.drops.no_mate entry 'Z' is not a declared piece" in errors, errors


def test_a_malformed_wrap_by_player_does_not_stop_the_later_axes() -> None:
    errors = validate_spec(
        _patched(BASE, {"rules.wrap": {"files_by_player": "left", "ranks": "x"}})
    )
    assert "rules.wrap.files_by_player must be an object keyed by '0' and '1'" in errors, errors
    assert any(e.startswith("rules.wrap.ranks must be one of") for e in errors), errors


def test_a_non_integer_board_side_stops_the_wall_checks() -> None:
    # The board check refuses the height; the wall check must not also try to read a
    # square against it (parse_square would compare a rank to the string '8').
    spec = _patched(BASE, {"board.height": "8", "rules.obstacles": {"squares": ["d4"]}})
    errors = validate_spec(spec)
    assert any(e.startswith("board.height") for e in errors), errors


def test_a_non_integer_board_side_stops_the_wild_route_checks() -> None:
    spec = _patched(
        BASE, {"board.height": "8", "rules.npcs": [{"piece": "R", "route": ["b2", "c2"]}]}
    )
    errors = validate_spec(spec)
    assert any(e.startswith("board.height") for e in errors), errors


def test_a_whole_valued_float_for_facing_axes_is_refused() -> None:
    # 4.0 == 4, so a bare membership test accepts it; the spelling has to be an integer.
    errors = validate_spec(_patched(BASE, {"pieces.1.facing": {"axes": 4.0}}))
    assert "pieces[1].facing.axes must be one of 2, 4" in errors, errors


def test_a_placement_for_player_zero_point_zero_still_holds_its_square() -> None:
    # 0.0 passes `player in (0, 1)`; under the default mirror it is filed with player 0,
    # and c1 is already the king's.
    spec = copy.deepcopy(BASE)
    spec["setup"]["placements"].append({"piece": "R", "player": 0.0, "squares": ["c1"]})
    errors = validate_spec(spec)
    assert "setup.placements[2].squares 'c1' is already occupied by 'K'" in errors, errors
