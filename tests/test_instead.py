"""The nearest allowed game to a refused one: `spec/instead.py` (R-L3, B2).

`docs/reference/refusal-offers.md` §4 is the design and §8 the tests this file holds: every
offer is allowed, states what was asked for, is not the draft it came from and is the trial
with exactly its edits applied (test 1); the first offer is no further than any single edit
that is allowed (test 2); the offers are the same when every sentence is replaced by an opaque
token (test 3, no prose); the walk is deterministic and bounded (test 5); and the validator's
answers are unmoved (test 6).

The twenty witnesses are refused drafts built from committed rulesets, each pinned with the
refusals it meets (`refusals.entry_of`) and its first offer. They cover the four kinds of §5,
read mechanically from each entry's ``pair``, and the three reasons, at least three of each.
Chess with rounds is built from chess because the design pins it (§8 test 2).
"""

from __future__ import annotations

import ast
import copy
import functools
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import cli
from chess_variant_sandbox.spec import instead, validate_spec
from chess_variant_sandbox.spec.instead import (
    BEAM,
    LEVELS,
    OFFERS,
    VALIDATIONS,
    Offer,
    apply_edits,
    edit_text,
    edits,
    nearest,
    search,
    states,
)
from chess_variant_sandbox.spec.refusals import _classification, entry_of
from chess_variant_sandbox.spec.relate import identity_key, units
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _piece(spec: dict[str, Any], pid: str) -> dict[str, Any]:
    piece: dict[str, Any] = next(p for p in spec["pieces"] if p["id"] == pid)
    return piece


# ------------------------------------------------------------------------- the witnesses

def _rounds(spec: dict[str, Any]) -> None:
    spec["rules"]["simultaneous"] = {}


def _rounds_by_capture(spec: dict[str, Any]) -> None:
    spec["rules"].update(simultaneous={}, win="royal_capture")
    spec["rules"].pop("en_passant", None)


def _passes(spec: dict[str, Any]) -> None:
    spec["rules"]["pass"] = True


def _hired_in_rounds(spec: dict[str, Any]) -> None:
    _rounds_by_capture(spec)
    spec["rules"].pop("castling")


def _shot_beside_choose(spec: dict[str, Any]) -> None:
    spec["rules"]["strike"] = {"position": "stay", "immune": ["K"]}


def _lopsided_blast(spec: dict[str, Any]) -> None:
    blast = spec["rules"]["explode_on_capture"]
    blast.pop("radius", None)
    blast["blast"] = {"vectors": [[0, 1]], "symmetry": "none"}


def _promoting_sweeper(spec: dict[str, Any]) -> None:
    _piece(spec, "P")["moves"].append(
        {"kind": "sweep", "vectors": [[0, 1]], "range": 2, "mode": "capture"},
    )


def _no_promotion(spec: dict[str, Any]) -> None:
    spec["rules"].pop("promotion")


def _knight_duck(spec: dict[str, Any]) -> None:
    spec["rules"]["duck"] = {"piece": "N"}


def _commands_the_king(spec: dict[str, Any]) -> None:
    _piece(spec, "B")["sight"][0]["affects"] = ["P", "K"]


def _king_for_nobody(spec: dict[str, Any]) -> None:
    spec["setup"]["placements"].append({"piece": "K", "player": "nobody", "squares": ["c3"]})


def _set(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def made(spec: dict[str, Any]) -> None:
        spec["rules"][key] = value
    return made


def _orientation(spec: dict[str, Any]) -> None:
    spec["setup"]["orientation"] = "mirror"


# name: (file, how the draft is refused, wanted, from the file (a base), the entries met, in
# the validator's order (None: a problem with no entry), its own entry, the first offer's edits)
Witness = tuple[str, Callable[[dict[str, Any]], None], str | None, bool, list[str | None], str,
                list[str] | None]
WITNESSES: dict[str, Witness] = {
    "chess in rounds": (
        "rulesets/chess.json", _rounds, "rules.simultaneous", True, ["R1", "R7"], "R1",
        ['set rules.win = "royal_capture"', "drop rules.en_passant"],
    ),
    "a pass beside portals": (
        "archive/ferry.json", _passes, "rules.pass", True, ["pass.P7"], "pass.P7",
        ["drop rules.portals"],
    ),
    "a pass where a capture is a must": (
        "rulesets/antichess.json", _passes, "rules.pass", True, [None, "pass.P4"], "pass.P4",
        ['set rules.stalemate = "draw"', "drop rules.must_capture"],
    ),
    "turn parts in rounds": (
        "rulesets/drover.json", _rounds, "rules.simultaneous", True,
        ["L4", "shared.rounds", "parts.rounds"], "parts.rounds",
        ["drop rules.turn", "drop pieces[1].control"],
    ),
    "the duck beside forbidden check": (
        "rulesets/duck-chess.json", _set("forbid_check", True), "rules.forbid_check", True,
        ["K1.forbid_check"], "K1.forbid_check", ["drop rules.duck"],
    ),
    "a hex army mirrored": (
        "rulesets/glinski-hexagonal-chess-with-mirrored.json", _orientation, None, False,
        ["X5", None], "X5", ['set setup.orientation = "rotate"'],
    ),
    "the duck's clock stopping on its half": (
        "rulesets/duck-chess.json", _set("max_plies", 801), None, False, ["K4"], "K4",
        ["drop rules.duck"],
    ),
    "hiring in rounds": (
        "archive/the-free-companies.json", _hired_in_rounds, "rules.simultaneous", True,
        ["R19.hire"], "R19.hire",
        ["drop pieces[6].price", "drop rules.purse", "drop pieces[].bounty"],
    ),
    "a straight-back bar in rounds": (
        "archive/ferry-no-straight-back.json", _rounds_by_capture, "rules.simultaneous", True,
        ["R9.bar"], "R9.bar", ["drop rules.portals[].no_straight_back"],
    ),
    "banking goals in rounds": (
        "rulesets/crossings.json", _rounds, "rules.simultaneous", True,
        ["goals.rounds"], "goals.rounds", ["drop rules.reach"],
    ),
    "a chosen strike in rounds": (
        "archive/hierarchy-rise-throw.json", _rounds_by_capture, "rules.simultaneous", True,
        ["R10.strike"], "R10.strike", ["drop rules.strike"],
    ),
    "a chosen contact beside a strike": (
        "archive/pawn-knockdown.json", _shot_beside_choose, "rules.strike", True,
        ["T3.choose"], "T3.choose", ["drop rules.contacts"],
    ),
    "a blast that reads differently for the two": (
        "rulesets/atomic.json", _lopsided_blast, None, False, ["blast.frame"], "blast.frame",
        ["drop rules.explode_on_capture.blast"],
    ),
    "a promoting sweeper beside folds": (
        "archive/crease.json", _promoting_sweeper, None, False, ["W7.fold"], "W7.fold",
        ["drop rules.portals[].kind"],
    ),
    "en passant with no promotion": (
        "rulesets/berolina.json", _no_promotion, None, False, ["H12"], "H12",
        ["drop rules.en_passant"],
    ),
    "annihilation beside a royal": (
        "rulesets/gardner.json", _set("win", "annihilation"), None, False, ["Z3"], "Z3",
        ['set rules.win = "checkmate"'],
    ),
    "a knight for a duck": (
        "rulesets/gardner.json", _knight_duck, "rules.duck", True,
        ["K1.win", None, "K11", "K11"], "K11", None,
    ),
    "a command over the king": (
        "archive/chess-with-preaching-bishops.json", _commands_the_king, None, False,
        ["V10"], "V10", ["drop pieces[2].sight"],
    ),
    "a royal nobody owns": (
        "rulesets/gardner.json", _king_for_nobody, None, False, ["O5"], "O5",
        ["drop pieces[5].royal", 'set rules.win = "annihilation"'],
    ),
    "a pass beside a push": (
        "rulesets/dohyo.json", _passes, "rules.pass", True, ["pass.P8"], "pass.P8", None,
    ),
}


def _made(name: str) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    """The witness's trial, wanted and base, built afresh."""
    path, refuse, wanted, from_base, *_ = WITNESSES[name]
    base = _load(path)
    trial = copy.deepcopy(base)
    refuse(trial)
    return trial, wanted, (base if from_base else None)


@functools.cache
def _found(name: str) -> instead.Search:
    trial, wanted, base = _made(name)
    return search(trial, wanted=wanted, base=base)


def _kind(pair: tuple[str, str] | list[str]) -> str:
    """§5's kind, read mechanically from how ``pair`` spells the two halves: a half that is no
    key path is a class or a limit; else one that names a value is a value; else one with a
    narrowing in words is a narrower case; else two whole rules."""
    halves = []
    for half in pair:
        head = half.split(" ")[0].split(":")[0]
        if not head.startswith(("rules.", "pieces[]", "setup.", "board.")):
            halves.append("class")
            continue
        rest = half[len(head):]
        halves.append("whole" if rest == "" else "value" if rest.startswith(":") else "narrower")
    for kind in ("class", "value", "narrower"):
        if kind in halves:
            return kind
    return "whole"


# ------------------------------------------------------------------------------ coverage

def test_the_witnesses_cover_every_kind_and_reason() -> None:
    assert len(WITNESSES) == 20
    raw = {str(entry["id"]): entry for entry in _classification()}
    kinds: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for _name, (*_, own, _first) in WITNESSES.items():
        kinds[_kind(raw[own]["pair"])] = kinds.get(_kind(raw[own]["pair"]), 0) + 1
        reasons[raw[own]["reason"]] = reasons.get(raw[own]["reason"], 0) + 1
    assert set(kinds) == {"whole", "value", "narrower", "class"} and min(kinds.values()) >= 3, kinds
    assert set(reasons) == {"contradictory", "too_expensive", "not_built"}, reasons
    assert min(reasons.values()) >= 3, reasons


@pytest.mark.parametrize("name", list(WITNESSES))
def test_each_witness_meets_its_refusals(name: str) -> None:
    """§8 test 6 for the witnesses: what the validator says of each is pinned, as entries."""
    trial, _wanted, _base = _made(name)
    met = [getattr(entry_of(problem), "id", None) for problem in validate_spec(trial)]
    assert met == WITNESSES[name][4]
    assert WITNESSES[name][5] in met


# ---------------------------------------------------------------------- 1: every offer

def _assert_offer(trial: dict[str, Any], offer: Offer, wanted: str | None,
                  base: dict[str, Any] | None) -> None:
    assert validate_spec(offer.draft) == [], offer.edits
    assert states(offer.draft, wanted)
    if base is not None:
        assert identity_key(offer.draft) != identity_key(base)
    assert apply_edits(trial, offer.edits) == offer.draft
    assert offer.units == units(trial, offer.draft)


@pytest.mark.parametrize("name", list(WITNESSES))
def test_every_offer_is_allowed_and_is_its_edits(name: str) -> None:
    trial, wanted, base = _made(name)
    offers = _found(name).offers
    assert len(offers) <= OFFERS
    assert len({identity_key(offer.draft) for offer in offers}) == len(offers)
    for offer in offers:
        _assert_offer(trial, offer, wanted, base)


# Rules a designer's menu offers, at the values a card arrives holding for a chess-like draft
# (`tests/test_designer_coverage.py`'s set, and the pass and a compelled capture).
MENU: dict[str, Any] = {
    "simultaneous": {}, "castling": {"king": "K", "rook": "R"}, "en_passant": True,
    "forbid_check": True, "turn": {"moves": 2}, "pass": True, "must_capture": True,
}


@pytest.mark.slow
@pytest.mark.parametrize("path", sorted((ROOT / "rulesets").glob("*.json")), ids=lambda p: p.stem)
def test_every_offer_over_the_library_and_the_menu_is_allowed(path: Path) -> None:
    """§8 test 1 over every committed ruleset, each rule of the menu tried on it. About 20 s
    in all, three of it on the shogis' en passant (1,211 validations)."""
    base = json.loads(path.read_text(encoding="utf-8"))
    assert validate_spec(base) == []
    for key, value in MENU.items():
        if key in base["rules"]:
            continue
        trial = copy.deepcopy(base)
        trial["rules"][key] = value
        if not validate_spec(trial):
            continue
        found = search(trial, wanted=f"rules.{key}", base=base)
        assert found.validations <= VALIDATIONS
        for offer in found.offers:
            _assert_offer(trial, offer, f"rules.{key}", base)


# ------------------------------------------------------------------------- 2: nearest

@pytest.mark.parametrize("name", list(WITNESSES))
def test_the_first_offer_is_pinned_and_nearest(name: str) -> None:
    trial, wanted, base = _made(name)
    offers = _found(name).offers
    first = WITNESSES[name][6]
    if first is None:
        assert offers == ()
        return
    assert [edit_text(edit) for edit in offers[0].edits] == first
    assert [o.units for o in offers] == sorted(o.units for o in offers)
    # No single edit of §4.1 gives an allowed draft (stating wanted, not base) that is nearer.
    problems = frozenset(validate_spec(trial))
    base_key = identity_key(base) if base is not None else None
    for _cls, edit in edits(current_spelling(trial), problems, wanted):
        draft = apply_edits(trial, (edit,))
        if draft is None or not states(draft, wanted) or validate_spec(draft):
            continue
        if base_key is not None and identity_key(draft) == base_key:
            continue
        assert units(trial, draft) >= offers[0].units, edit


# ------------------------------------------------------------------------- 3: no prose

def _opaque(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each validator sentence replaced by a token that keeps only its identity, and every
    entry lookup by the sentence it stands for: the search reads no sentence."""
    real_validate = validate_spec
    real_entry = instead.entry_of
    sentences: dict[str, str] = {}

    def tokens(spec: Any) -> list[str]:
        out = []
        for sentence in real_validate(spec):
            token = hashlib.sha256(sentence.encode()).hexdigest()
            sentences[token] = sentence
            out.append(token)
        return out

    monkeypatch.setattr(instead, "validate_spec", tokens)
    monkeypatch.setattr(instead, "entry_of", lambda token: real_entry(sentences.get(token, "")))


def _said(offers: tuple[Offer, ...] | list[Offer]) -> list[tuple[Any, ...]]:
    return [(o.edits, o.units, json.dumps(o.draft, sort_keys=True)) for o in offers]


_QUICK = ("chess in rounds", "hiring in rounds", "a straight-back bar in rounds",
          "a royal nobody owns", "a blast that reads differently for the two")


@pytest.mark.parametrize("name", _QUICK)
def test_the_offers_are_the_same_with_every_sentence_a_token(
    name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _said(_found(name).offers)
    _opaque(monkeypatch)
    trial, wanted, base = _made(name)
    assert _said(nearest(trial, wanted=wanted, base=base)) == expected


@pytest.mark.slow
@pytest.mark.parametrize("name", [name for name in WITNESSES if name not in _QUICK])
def test_the_offers_are_the_same_with_every_sentence_a_token_for_every_witness(
    name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_the_offers_are_the_same_with_every_sentence_a_token(name, monkeypatch)


# ---------------------------------------------------------- 5: deterministic, bounded

# What the walk costs from chess with rounds (§4.2 step 4): a ceiling that may only fall.
CHESS_IN_ROUNDS_VALIDATIONS = 123


def test_the_constants_are_pinned() -> None:
    assert (LEVELS, BEAM, VALIDATIONS, OFFERS) == (3, 4, 2_000, 3)
    assert instead.CLASSES == ("V", "S", "P", "R", "D")


def test_the_walk_is_deterministic_and_bounded() -> None:
    trial, wanted, base = _made("chess in rounds")
    once = search(trial, wanted=wanted, base=base)
    again = search(copy.deepcopy(trial), wanted=wanted, base=base)
    assert _said(once.offers) == _said(again.offers)
    assert once.validations <= CHESS_IN_ROUNDS_VALIDATIONS
    assert once.levels == 2
    assert all(found.validations <= VALIDATIONS for found in map(_found, WITNESSES))


def test_an_allowed_draft_has_no_offers() -> None:
    gardner = _load("rulesets/gardner.json")
    assert nearest(gardner) == []
    assert search(gardner).validations == 1


def test_a_draft_refused_only_in_its_spelling_is_offered_tidied() -> None:
    """A rule stated at its off value is its absence: the tidy alone is the offer."""
    trial = _load("rulesets/gardner.json")
    trial["rules"]["simultaneous"] = {"tie": "both"}
    trial["rules"]["win"] = "royal_capture"
    if validate_spec(trial):
        offers = nearest(trial)
        assert offers and validate_spec(offers[0].draft) == []


# ------------------------------------------------------------------- the module itself

def test_the_module_reads_no_sentence_and_nothing_of_the_web() -> None:
    tree = ast.parse(Path(instead.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    assert "re" not in names
    assert not any(name.split(".")[:2] == ["chess_variant_sandbox", "web"] for name in names)


def test_the_path_spelling_round_trips() -> None:
    for text in ("rules.win", "pieces[2].moves[0].push", "pieces[].moves[].mode",
                 "rules.portals[].no_straight_back", "board.lattice"):
        assert instead.path_text(instead.parse_path(text)) == text


def test_no_edit_leaves_out_what_was_asked_for() -> None:
    trial, wanted, _base = _made("chess in rounds")
    assert wanted is not None
    for _cls, edit in edits(trial, frozenset(), wanted):
        assert edit[1] not in ("rules", wanted)


# ---------------------------------------------------------------------- the command line

def test_validate_prints_the_offers_with_instead(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    trial, _wanted, _base = _made("a straight-back bar in rounds")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(trial), encoding="utf-8")
    assert cli.main(["validate", str(path)]) == 1
    plain = capsys.readouterr().out
    assert "instead" not in plain
    assert cli.main(["validate", str(path), "--instead"]) == 1
    out = capsys.readouterr().out
    assert out.startswith(plain)
    offers = nearest(trial)
    lines = out[len(plain):].splitlines()
    assert lines[0] == "instead, the nearest allowed:"
    assert lines[1:] == [
        f"  - {o.units} rule{'' if o.units == 1 else 's'} away: "
        + "; ".join(edit_text(e) for e in o.edits)
        for o in offers
    ]
