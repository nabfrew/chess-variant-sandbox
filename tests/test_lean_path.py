"""R-L5c2: the core's common path compiled twice, and what holds the two copies to one game.

`docs/reference/common-path-plan.md` §2.5. A game with no rare rule plays the *lean* copy of
move generation (`rust/src/path.rs`): the full copy's text with every rare family's gate
compiled out. Two things can make them differ, and each has a test here:

* a gate that reads a field `Tables::rare_in_use` does not read, so a game using that rule
  would be called lean and play a copy without its arm (item 2: the source reading below);
* anything else, which the forced-full switch catches: every stored game the core calls lean
  is played on both copies and must count, walk and play identically (item 4).

The fuzz's `the_lean_copy_offers_what_the_full_copy_offers` is item 5; a checking build's
`debug_assert!` in every gate is item 3.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import _core
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec

ROOT = Path(__file__).resolve().parent.parent
RUST_SRC = ROOT / "rust" / "src"
STORED = sorted([*(ROOT / "rulesets").glob("*.json"), *(ROOT / "archive").glob("*.json")])

# ------------------------------------------------------------- item 2: the gates' fields

# A gate that names no `t.<field>` names what it reads some other way; each is listed with
# the `Tables` fields behind it. A new spelling fails `test_every_gate_reads_a_field_rare_in_use_reads`
# until it is listed here.
GATE_METHODS = {"duck_owed": {"duck"}, "royal_in_hand": {"drops"}}
GATE_FUNCTIONS = {"enters_play": {"drops", "purse"}}
GATE_WORDS = {"fan": {"strike"}, "MODE_PUSH": {"push"}}

_GATE = re.compile(r"\b(rare!|rare_opt|decode_owned_if)\(")


def _arguments(text: str, start: int) -> tuple[list[str], int]:
    """The top-level comma-separated arguments of the call whose `(` is at ``start``."""
    depth, args, current = 0, [], []
    for index in range(start, len(text)):
        char = text[index]
        if char in "([{":
            depth += 1
            if depth == 1:
                continue
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(current).strip())
                return args, index
        elif char == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    raise ValueError("an unclosed gate")


def _strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def gates(source: str) -> list[tuple[str, str, str]]:
    """Every gate in ``source``: (macro or function, family argument, tested expression)."""
    text = _strip_comments(source)
    found = []
    for match in _GATE.finditer(text):
        name = match.group(1)
        if text[: match.start()].rstrip().endswith("fn"):
            continue  # the definition
        args, _ = _arguments(text, match.end() - 1)
        if name == "decode_owned_if":
            found.append((name, args[0], "t.has_facing"))
        elif len(args) == 2 and "P::" in args[0]:
            found.append((name, args[0], args[1]))
    return found


def gate_fields(expression: str) -> set[str] | None:
    """The `Tables` fields a gate's expression reads, ``None`` if it names none we know."""
    fields = set(re.findall(r"\bt\.(\w+)", expression))
    for method in re.findall(r"\bself\.(\w+)\(", expression):
        if method not in GATE_METHODS:
            return None
        fields |= GATE_METHODS[method]
    for word, behind in {**GATE_FUNCTIONS, **GATE_WORDS}.items():
        if re.search(rf"\b{word}\b", expression):
            fields |= behind
    return fields or None


def rare_in_use_reads(tables_source: str) -> set[str]:
    text = _strip_comments(tables_source)
    start = text.index("pub fn rare_in_use(t: &Tables) -> bool {")
    depth, end = 0, len(text)
    for index in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            end = index
            break
    return set(re.findall(r"\bt\.(\w+)", text[start:end]))


def unread_gates(sources: dict[str, str], tables_source: str) -> list[str]:
    """Each gate whose fields `rare_in_use` does not all read, or whose expression names none."""
    read = rare_in_use_reads(tables_source)
    problems = []
    for name, source in sources.items():
        for kind, family, expression in gates(source):
            fields = gate_fields(expression)
            if fields is None:
                problems.append(f"{name}: {kind}({family}, {expression}) names no field this test knows")
            elif fields - read:
                problems.append(f"{name}: {kind}({family}, {expression}) reads {sorted(fields - read)}")
    return problems


def _rust_sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(RUST_SRC.glob("*.rs"))}


def test_every_gate_reads_a_field_rare_in_use_reads() -> None:
    sources = _rust_sources()
    assert len([g for s in sources.values() for g in gates(s)]) >= 30, "the gates moved; reading the wrong tree"
    assert unread_gates(sources, sources["tables.rs"]) == []


def test_the_reading_fails_on_a_field_rare_in_use_forgets() -> None:
    """The planted fault: `has_sweep` struck from `rare_in_use`."""
    sources = _rust_sources()
    forgetful = sources["tables.rs"].replace("|| t.has_sweep\n", "\n", 1)
    assert forgetful != sources["tables.rs"]
    problems = unread_gates(sources, forgetful)
    assert problems and all("has_sweep" in p for p in problems)


def test_the_reading_fails_on_a_gate_it_cannot_read() -> None:
    planted = {"x.rs": "if rare!(P::SWEEP, self.something_new()) {}"}
    assert "names no field" in unread_gates(planted, _rust_sources()["tables.rs"])[0]


# ------------------------------------------------------------ item 4: forced-full equality

@contextmanager
def _full_copy() -> Iterator[None]:
    before = _core.force_full_path(True)
    try:
        yield
    finally:
        _core.force_full_path(before)


def _pair(path: Path) -> tuple[Engine, Engine]:
    spec = load_spec(path)
    lean = Engine(spec)
    with _full_copy():
        full = Engine(spec)
    assert lean.core.lean_path() and not full.core.lean_path()
    return lean, full


def _lean_files() -> list[Path]:
    return [path for path in STORED if Engine(load_spec(path)).core.lean_path()]


LEAN = _lean_files()


def test_chess_and_a_third_of_the_corpus_play_the_lean_copy() -> None:
    names = {path.stem for path in LEAN}
    assert "chess" in names
    assert len(LEAN) >= 30, len(LEAN)
    assert "crazyhouse" not in names and "atomic" not in names


def test_the_switch_is_read_when_the_tables_are_built() -> None:
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    with _full_copy():
        full = Engine(spec)
    assert not full.core.lean_path()
    assert Engine(spec).core.lean_path()


@pytest.mark.parametrize("path", LEAN, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_a_lean_game_counts_alike_on_both_copies(path: Path) -> None:
    lean, full = _pair(path)
    state = lean.initial_state()
    assert lean.perft(state, 3) == full.perft(state, 3)


@pytest.mark.parametrize("path", LEAN, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_a_lean_game_walks_alike_on_both_copies(path: Path) -> None:
    """`test_schema_two.py`'s walk, every position's list and key read off both copies."""
    import random

    lean, full = _pair(path)
    rng = random.Random(7)
    state = lean.initial_state()
    for _ in range(60):
        moves = lean.legal_moves(state)
        assert moves == full.legal_moves(state)
        assert lean.terminal(state, moves) == full.terminal(state, moves)
        if lean.terminal(state, moves) is not None:
            break
        move = rng.choice(moves)
        after = lean.apply(state, move)
        assert after == full.apply(state, move)
        assert after.key == lean._compute_hash(after)
        state = after


@pytest.mark.slow
@pytest.mark.parametrize("path", LEAN, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_a_lean_game_counts_alike_on_both_copies_deeper(path: Path) -> None:
    lean, full = _pair(path)
    state = lean.initial_state()
    assert lean.perft(state, 4) == full.perft(state, 4)


_WALL_CLOCK = re.compile(rb',"duration_ms":[^,}]*')


def _records(spec: dict[str, Any], out: Path) -> bytes:
    from chess_variant_sandbox.tournament import TournamentConfig, run_tournament
    from chess_variant_sandbox.tournament.record import write_records

    config = TournamentConfig(agents=("random", "greedy", "minimax:nodes=400"), games_per_pairing=2,
                              seed=0, workers=1)
    write_records(run_tournament(spec, config).records, out)
    return _WALL_CLOCK.sub(b"", out.read_bytes())


def test_a_seeded_tournament_is_byte_identical_on_both_copies(tmp_path: Path) -> None:
    """One seeded tournament's records, the wall clock aside (`measurement.records_digest`'s
    reading), on a lean game played on each copy."""
    spec = load_spec(ROOT / "rulesets" / "gardner.json")
    assert Engine(spec).core.lean_path()
    lean = _records(spec, tmp_path / "lean.jsonl")
    with _full_copy():
        full = _records(spec, tmp_path / "full.jsonl")
    assert lean and lean == full
