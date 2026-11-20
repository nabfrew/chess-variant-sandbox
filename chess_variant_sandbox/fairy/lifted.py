"""How Fairy-Stockfish treats each pair the rule language has lifted (Q1218).

`spec/refusals.toml`'s `[[lifted]]` rows are pairs the validator once refused and now plays.
Fairy-Stockfish's refusals in `variant.py` were written pair by pair as the language grew, and
a lift can hand the translator a pair it was never asked about: `gating.duck` (U2) translated,
and Fairy-Stockfish offered no gate beside the duck. This module asks, for every row, whether
`fairy/` can express a small game holding the pair, and where it says yes, whether `verify`
holds the two engines' legal moves together.

The games are the fuzz package's (`fuzz_lifted.generate.case`, one row at a time), so a pair
is built the way the soak builds it, with the six-by-six parent's first-move double step taken
out where the pair survives it (`_without_first_moves`), and hand-built games (`HAND`) for the
rows no maker reaches and for a translation the small parents cannot show. The verdict per row
is one of:

* ``refused``: every case built is refused by the translator.
* ``verified``: some case translates, and every case that translates verifies.
* ``diverges``: some case translates and `verify` finds a position the engines disagree on.
* ``unverified``: some case translates and no `pyffish` is here to verify it.
* ``unbuilt``: no maker reaches the pair, so nothing was asked.

`TREATMENT` is the table the fast suite holds every `[[lifted]]` row to
(`tests/test_fairy_lifted.py`): a new lifted row without an entry fails, so the next lift is
put to Fairy-Stockfish before it can be forgotten. The probe that fills it is

    uv run python -m chess_variant_sandbox.fairy.lifted [--rows ID ...] [--json out.json]

and its last table is in `docs/reference/fairy-stockfish.md` (Q1218).
"""

# compute-lease: light

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# The verify the probe runs: the record's depth (3, `verify.DEFAULT_DEPTH` and the CLI's),
# a shorter walk than the CLI's 400 plies, since the cases are 5x5 and 6x6 games.
DEPTH = 3
GAMES = 2
PLIES = 80
# How many cases a row is probed with, and how many seeds it may take to build them.
CASES = 3
SEEDS = 40

VERDICTS = ("refused", "verified", "diverges", "unverified", "unbuilt")

# Every `[[lifted]]` row, and how Fairy-Stockfish treats it: the verdict and its evidence.
# Filled from the probe (Q1218, pyffish 0.0.90, 2026-11-14); `docs/reference/fairy-stockfish.md`
# has the table. A `refused` row's evidence is the refusal that names the pair, alternatives
# split by " | ": a reason one of its two keys gets on its own whatever stands beside it, or
# the pair's own (`rules.drops.gating beside rules.duck`). `check` holds every refused case
# of the row to it: each is refused, and one at least by a reason holding an alternative (the
# rest may stop earlier, at a parent's piece the notation cannot write). A `verified` row's
# evidence says which game verified. A row the probe reads as `diverges` gets no entry here:
# it gets a refusal in `variant.py` first, and is then `refused`.
TREATMENT: dict[str, tuple[str, str]] = {
    "L1": ("refused", "rules.turn: a side moves once per turn"),
    "L2": ("refused", "pieces[].rest: | rules.turn: a side moves once per turn"),
    "L3": (
        "refused", "rules.npcs: a piece neither player owns | rules.turn: a side moves once per turn",
    ),
    "L5": (
        "refused", "pieces.control: a piece moved by whichever side | rules.turn: a side moves once per turn",
    ),
    "L6": ("refused", "rules.purse: | rules.turn: a side moves once per turn"),
    "L7": ("refused", "rules.duck beside rules.turn"),
    "R6": ("refused", "rules.simultaneous:"),
    "R9": ("refused", "rules.simultaneous: | rules.portals:"),
    "N4": ("refused", "rules.npcs:"),
    "N5": ("refused", "rules.npcs:"),
    "N5.forbid_check": ("refused", "rules.npcs:"),
    "N5.safe": ("refused", "rules.npcs:"),
    "N7": ("refused", "rules.npcs:"),
    "R11": ("refused", "rules.simultaneous: | rules.strike:"),
    "R18": ("refused", "rules.simultaneous: | setup.placements.player: a piece nobody owns"),
    "R19": ("refused", "rules.simultaneous: | rules.purse:"),
    "R20": ("refused", "rules.simultaneous: | pieces.control: a piece moved by whichever side"),
    "T1": ("refused", "rules.strike:"),
    "T2": ("refused", "rules.strike:"),
    "T4": ("refused", "rules.contacts:"),
    "T5": ("refused", "rules.contacts:"),
    "T6": ("refused", "rules.contacts: | rules.npcs: a piece neither player owns"),
    "T7": ("refused", "rules.strike: | rules.npcs: a piece neither player owns"),
    "T8": ("refused", "rules.npcs: a piece neither player owns"),
    "T9": ("refused", "rules.contacts:"),
    "T10": ("refused", "rules.strike:"),
    "H1": ("verified",
        "a stock hand verified to depth 3 (pyffish 0.0.90); a capture hand is refused "
        "by name: rules.promotion.only_lost beside rules.drops filling from captures"
    ),
    "H2": ("refused", "rules.strike:"),
    "H3": ("refused", "rules.promote_on_entry:"),
    "H6": ("refused", "rules.drops.either_face beside rules.explode_on_capture"),
    "H11": ("refused", 'rules.extinction.per_type "own"'),
    "W1": ("refused", "a sweep captures what it passes over | rules.strike:"),
    "W2": ("refused", "a sweep captures what it passes over | rules.contacts:"),
    "W3": ("refused", "a sweep captures what it passes over"),
    "W4": ("refused", "a sweep captures what it passes over"),
    "W5": (
        "refused", "a sweep captures what it passes over | rules.npcs: a piece neither player owns",
    ),
    "W6": ("refused", "a sweep captures what it passes over | rules.wrap:"),
    "W7": ("refused", "a sweep captures what it passes over"),
    "W8": ("refused", "a sweep captures what it passes over | rules.portals:"),
    "W9": ("refused", "a sweep captures what it passes over"),
    "W10": ("refused", "pieces[].sight: | a sweep captures what it passes over"),
    "V1": ("refused", "pieces[].sight: | rules.portals:"),
    "V2": ("refused", "pieces[].sight:"),
    "V3": ("refused", "pieces[].sight:"),
    "V4": ("refused", "pieces[].sight: | rules.territory:"),
    "V5": ("refused", "pieces[].sight: | rules.strike:"),
    "V6": ("refused", "pieces[].sight: | rules.contacts:"),
    "V7": ("refused", "pieces[].sight: | pieces[].rest:"),
    "V8": ("refused", "pieces[].sight: | a sweep captures what it passes over"),
    "V9": ("refused", "pieces[].sight:"),
    "V11": ("refused", "pieces[].sight:"),
    "V15": ("refused", "pieces[].sight: | setup.placements.player: a piece nobody owns"),
    "N17": ("refused", "rules.npcs:"),
    "N20": ("refused", "path turns as it goes | rules.npcs:"),
    "K2": ("refused", "rules.portals:"),
    "K3": (
        "refused", "rules.duck beside rules.explode_on_capture | rules.strike: | rules.contacts:",
    ),
    "P1": ("refused", "rules.portals:"),
    "P3": ("refused", "rules.portals: | rules.strike:"),
    "P4": ("refused", "rules.portals: | rules.contacts:"),
    "P5": ("refused", "rules.portals:"),
    "P6": ("refused", "rules.portals: | rules.territory:"),
    "P7": ("refused", "rules.portals:"),
    "P9": ("refused", "rules.purse: | rules.portals:"),
    "P10": ("refused", "setup.placements.player: a piece nobody owns | rules.portals:"),
    "O1": (
        "refused", "setup.placements.player: a piece nobody owns | rules.npcs: a piece neither player owns",
    ),
    "O2": ("refused", "setup.placements.player: a piece nobody owns"),
    "O3": ("refused", "setup.placements.player: a piece nobody owns | rules.strike:"),
    "O4": ("refused", "setup.placements.player: a piece nobody owns | rules.contacts:"),
    "O6": ("refused", "setup.placements.player: a piece nobody owns | pieces[].spy:"),
    "O9": ("refused", "setup.placements.player: a piece nobody owns | piece.facing:"),
    "O10": ("refused", "pieces.control: a piece moved by whichever side | rules.territory:"),
    "O11": ("refused", "pieces.control: a piece moved by whichever side"),
    "O12": ("refused", "pieces.control: a piece moved by whichever side"),
    "O13": ("refused", "pieces.control: | piece.facing:"),
    "O14": ("refused", "pieces.control:"),
    "O15": ("refused", "pieces.control: a piece moved by whichever side"),
    "O16": ("refused", "setup.placements.player: a piece nobody owns"),
    "O17": ("refused", "pieces.control:"),
    "O18": ("refused", "pieces.control:"),
    "O19": ("refused", "pieces.control:"),
    "O22": ("refused", "rules.purse:"),
    "O24": ("refused", "rules.purse: | pieces[].spy:"),
    "O27": ("refused", "rules.purse: | pieces[].spy:"),
    "Y3": ("refused", "pieces[].spy:"),
    "Y4": ("refused", "rules.strike: | pieces[].spy:"),
    "Y5": ("refused", "rules.strike: | pieces[].spy:"),
    "Y6": ("refused", "rules.contacts: | pieces[].spy:"),
    "Y7": ("refused", "pieces[].spy:"),
    "Y8": ("refused", "pieces[].spy:"),
    "PA2": ("refused", "path turns as it goes"),
    "PA4": ("refused", "path turns as it goes | a lame rider"),
    "X1": ("verified",
        "a hex lame whose legs are Fairy-Stockfish's own (nD, rotate6) verified to "
        "depth 3 (pyffish 0.0.90); other legs are refused by name: a lame leap blocked"
    ),
    "X8": ("verified",
        "a step leap under blocks_leaps verified to depth 3 (pyffish 0.0.90); a longer "
        "leap a wall would stop is refused by name: rules.obstacles.blocks_leaps"
    ),
    "Z2": ("refused", "castling with a piece that is not this ruleset's royalty"),
    "push.N5": ("refused", "pieces[].moves[].push: | pieces[].rest:"),
    "push.G1": ("refused", "pieces[].moves[].push:"),
    "push.G2": ("refused", "pieces[].moves[].push: | rules.territory:"),
    "push.purse": ("refused", "pieces[].moves[].push: | rules.purse:"),
    "push.sight": ("refused", "pieces[].moves[].push: | pieces[].sight:"),
    "income.I1": ("refused", "rules.income: | rules.scoring:"),
    "gating.duck": ("refused", "rules.drops.gating beside rules.duck"),
    "gating.control": ("refused", "pieces.control:"),
    "goals.extinction": ("refused", "rules.reach removes or count"),
    "goals.bare_royal": ("refused", "rules.reach removes or count | rules.bare_royal:"),
    "shared.price": ("refused", "pieces.control: a piece nobody owns"),
    "run.territory": ("refused", "a hop over a whole run of pieces | rules.territory:"),
    "run.en_passant": ("refused", "a hop over a whole run of pieces"),
    # R-L4b stage 3 (probed 2026-11-18, pyffish 0.0.90): the capture list's pairs. A pair only
    # the list plays, as old keys or as a list only `rules.capture` says, is refused by name
    # (`variant._capture_list_reasons`, R-L4b4); a strike or contacts is refused on its own
    # besides.
    **{
        row: ("refused", "rules.capture:")
        for row in (
            "T1.position", "T1.choose", "T4.choose", "T2.keeping", "T5.keeping", "T3", "T4.switch",
            "H4", "H5", "W3.both", "push.X2.strike", "push.X2.contacts", "push.X3", "gating.blast",
            "gating.captures", "upkeep.U9", "upkeep.U10",
        )
    },
    "W1.disagree": ("refused", "rules.capture: | rules.strike:"),
    # R-L4b5: a round walks the capture list; Fairy-Stockfish refuses a round on its own.
    "T1.rounds": ("refused", "rules.simultaneous:"),
    "T4.rounds": ("refused", "rules.simultaneous:"),
}


@dataclass
class CaseResult:
    seed: int
    parent: str
    makers: tuple[str, ...]
    verdict: str  # refused | verified | diverges | unverified
    detail: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass
class RowResult:
    id: str
    pair: tuple[str, str]
    by: str
    verdict: str
    cases: list[CaseResult] = field(default_factory=list)


def _backend() -> Any | None:
    try:
        from chess_variant_sandbox.fairy.verify import PyffishBackend

        return PyffishBackend()
    except ImportError:
        return None


def _row_verdict(cases: list[CaseResult]) -> str:
    if not cases:
        return "unbuilt"
    said = {c.verdict for c in cases}
    if "diverges" in said:
        return "diverges"
    if said == {"refused"}:
        return "refused"
    if "unverified" in said:
        return "unverified"
    return "verified"


def probe_case(spec: dict[str, Any], backend: Any | None) -> tuple[str, str, list[str]]:
    """``(verdict, detail, reasons)`` for one spec: refused with its reasons, or verified
    against ``backend`` (``None``: unverified)."""
    from chess_variant_sandbox import fairy
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate
    from chess_variant_sandbox.fairy.verify import verify

    try:
        variant = translate(spec)
    except Untranslatable as error:
        return "refused", "; ".join(error.reasons), list(error.reasons)
    if backend is None:
        return "unverified", "no pyffish here", []
    result = verify(
        spec, backend, fairy.config_path(variant), depth=DEPTH, games=GAMES, plies=PLIES,
    )
    if result.ok:
        return "verified", result.summary(), []
    ours, theirs = set(result.ours), set(result.theirs)
    return "diverges", (
        f"{result.summary()} at {result.fen}; ours alone {sorted(ours - theirs)[:4]}, "
        f"theirs alone {sorted(theirs - ours)[:4]}"
    ), []


def _without_first_moves(spec: dict[str, Any], pair: tuple[str, str]) -> dict[str, Any] | None:
    """``spec`` with every move under a first-move condition taken out, if it still holds the
    pair and validates; else ``None``.

    The fuzz package's six-by-six parent gives its pawn a first-move double step, which the
    translator refuses on its own (a FEN cannot carry our `unmoved` mask for a piece that does
    not castle). Left in, it would refuse every case built on that parent whatever the pair
    is, and the probe would learn nothing about the pair. Taking it out is another game that
    holds the same pair, never a reading of the pair.
    """
    from chess_variant_sandbox.fuzz_lifted.keys import holds
    from chess_variant_sandbox.spec.schema import validate_spec

    trimmed = copy.deepcopy(spec)
    changed = False
    for piece in trimmed.get("pieces", []):
        moves = piece.get("moves", [])
        kept = [m for m in moves if not (m.get("condition") or {}).get("first_move")]
        if kept and len(kept) != len(moves):
            piece["moves"] = kept
            changed = True
    if not changed or validate_spec(trimmed) or not all(holds(trimmed, k) for k in pair):
        return None
    return trimmed


def _hex(moves: list[dict[str, Any]], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    king = {"id": "K", "value": 0, "royal": True,
            "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate6"}]}
    return {
        "name": "fairy lifted hex", "board": {"width": 7, "height": 7, "lattice": "hex"},
        "pieces": [king, {"id": "N", "value": 3, "moves": moves}],
        "setup": {"mirror": False, "orientation": "rotate", "placements": [
            {"piece": "K", "player": 0, "squares": ["d2"]},
            {"piece": "N", "player": 0, "squares": ["c2"]},
            {"piece": "K", "player": 1, "squares": ["d6"]},
            {"piece": "N", "player": 1, "squares": ["e6"]}]},
        "rules": {"win": "checkmate", **(rules or {})},
    }


_WALL = {"obstacles": {"squares": ["a4"], "blocks_leaps": True}}

# Hand-built cases, for the rows no fuzz maker reaches (PA4, X1, X8) and for a translation the
# small parents cannot show: a hex lame whose legs are Fairy-Stockfish's own (`nD` on a hex
# rhombus), and a step leap under `blocks_leaps`, which no wall can stop.
HAND: dict[str, tuple[dict[str, Any], ...]] = {
    "PA4": ({
        "name": "fairy lifted pa4", "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
            {"id": "N", "value": 3, "moves": [
                {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8", "range": 2},
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "path": "zigzag"}]}],
        "setup": {"placements": [{"piece": "K", "squares": ["c1"]}, {"piece": "N", "squares": ["e1"]}]},
        "rules": {"win": "checkmate"},
    },),
    "X1": (
        _hex([{"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral12", "legs": [[[0, 1]]]}]),
        _hex([{"kind": "lame", "vectors": [[2, 0]], "symmetry": "rotate6", "legs": [[[1, 0]]]}]),
    ),
    "X8": (
        _hex([{"kind": "leap", "vectors": [[2, 0]], "symmetry": "rotate6", "legs": [[[1, 0]]]}], _WALL),
        _hex([{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate6"}], _WALL),
    ),
}


def probe_row(row: Any, backend: Any | None, cases: int = CASES, seeds: int = SEEDS) -> RowResult:
    from chess_variant_sandbox.fuzz_lifted.generate import buildable, case

    out = RowResult(row.id, row.pair, row.by, "unbuilt")
    for index, spec in enumerate(HAND.get(row.id, ())):
        verdict, detail, reasons = probe_case(spec, backend)
        out.cases.append(CaseResult(-1 - index, "hand", (), verdict, detail, reasons))
    if buildable((row,)):
        built_cases = 0
        for seed in range(seeds):
            built = case(seed, (row,))
            if built.spec is None:
                continue
            spec = _without_first_moves(built.spec, row.pair) or built.spec
            verdict, detail, reasons = probe_case(spec, backend)
            out.cases.append(CaseResult(
                seed, built.parent, built.makers + built.extras, verdict, detail, reasons,
            ))
            built_cases += 1
            if built_cases >= cases:
                break
    out.verdict = _row_verdict(out.cases)
    return out


def check(row: Any, backend: Any | None) -> list[str]:
    """What is wrong with ``TREATMENT``'s entry for ``row``, probed afresh; empty when it holds.

    A ``refused`` row is probed without a backend (translation only, milliseconds); a
    ``verified`` row needs ``backend``, and is reported as unchecked without one.
    """
    entry = TREATMENT.get(row.id)
    if entry is None:
        return [f"{row.id}: no entry in fairy.lifted.TREATMENT"]
    verdict, evidence = entry
    if verdict == "refused":
        result = probe_row(row, None)
        if not result.cases:
            return [f"{row.id}: no case built, so nothing holds the refusal"]
        problems = [
            f"{row.id}: seed {c.seed} is not refused ({c.detail[:200]})"
            for c in result.cases if c.verdict != "refused"
        ]
        names = [n.strip() for n in evidence.split("|")]
        if not any(n in r for c in result.cases for r in c.reasons for n in names):
            problems.append(f"{row.id}: no case is refused by {evidence!r}")
        return problems
    if verdict == "verified":
        if backend is None:
            return [f"{row.id}: verified needs pyffish to check"]
        result = probe_row(row, backend)
        if result.verdict != "verified":
            return [f"{row.id}: the probe now says {result.verdict}"]
        return []
    return [f"{row.id}: {verdict!r} is not a verdict TREATMENT keeps"]


def common_reasons(result: RowResult) -> list[str]:
    """The refusal reasons every refused case of ``result`` gives."""
    sets = [set(c.reasons) for c in result.cases if c.verdict == "refused"]
    return sorted(set.intersection(*sets)) if sets else []


def main(argv: list[str] | None = None) -> int:
    from chess_variant_sandbox.fuzz_lifted.keys import lifted

    parser = argparse.ArgumentParser(description="Put every lifted pair to Fairy-Stockfish.")
    parser.add_argument("--rows", nargs="*", default=None, help="row ids (default: all)")
    parser.add_argument("--cases", type=int, default=CASES)
    parser.add_argument("--json", type=Path, default=None, help="write the results here")
    args = parser.parse_args(argv)

    backend = _backend()
    print(f"verify against: {'pyffish' if backend is not None else 'nothing (no pyffish)'}")
    rows = [r for r in lifted() if args.rows is None or r.id in args.rows]
    results: list[RowResult] = []
    for row in rows:
        result = probe_row(row, backend, cases=args.cases)
        results.append(result)
        print(f"{row.id:18s} {result.verdict:10s} {len(result.cases)} cases", flush=True)
        for c in result.cases:
            if c.verdict != "refused":
                print(f"    seed {c.seed} {c.verdict}: {c.detail[:300]}", flush=True)
        for reason in common_reasons(result):
            print(f"    every case: {reason[:160]}", flush=True)
        if args.json is not None:
            args.json.write_text(json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    counts = {v: sum(r.verdict == v for r in results) for v in VERDICTS}
    print("totals: " + ", ".join(f"{v} {n}" for v, n in counts.items()))
    return 1 if counts["diverges"] else 0


if __name__ == "__main__":  # pragma: no cover - a probe runner
    sys.exit(main())
