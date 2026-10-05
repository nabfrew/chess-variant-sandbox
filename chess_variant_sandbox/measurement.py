"""What a score rests on, and whether it still holds.

A number in `runs/` is a function of five things: the spec, the roster it was played on,
the weights it was scored under, the code that played the games, and the code that
turned those games into components. Four of those were recorded nowhere, so a
measurement could not answer the only question a reader ever has about it -- *is this
still true?* -- and the harness answered "yes" by default, which is the one answer it
had no evidence for.

The cost of that default is on the record. `v6` and `v7` are the same sixteen components
with the same weights to the digit; what moved between them was the *agents*, and 58 of
58 corpus entries played differently. Nothing on disk distinguished a `v7` number from a
`v6` number taken under the same weights version. And at the small end it produced the
ordinary daily version of the same failure: a fresh tournament beside a year-old verdict,
rendered as one row, with the ladder from one and the gate from the other.

So a measurement carries a `Basis` and readers ask `stale_against` rather than trusting
what they find. The rule this module exists to enforce:

> **A measurement whose basis does not match the current one is not a current number. It
> is a stale artefact with a reason attached, and it is displayed as one:** its last
> number marked *provisional*, with the revision and date it came from, never adopted,
> re-stamped or compared as current (`docs/plan-evolving-score.md` §3, which replaced *not
> a number* on 2026-10-03; `web/library.Game.status`).

Four fingerprints rather than one, because the kinds of change need different repairs.
A change under `engine` -- the core that walks the tables, the agents, and the loop and
schedule that run them -- invalidates the games themselves and the tournament has to be
replayed. A change under `tables`
invalidates them too, but for *one ruleset* rather than for all of them. A change under
`metrics` leaves the games valid -- they are persisted records, and re-scoring them
without an engine is the invariant this project already relies on. And a change under
`analysis` sits between those: the analysis pass drives the engine over the persisted
games and writes `analysis.json`, which the scorer then reads as a record. The games
stand, the file does not, and the repair is to run the pass again before re-scoring.
`Staleness.needs` says which, and `corpus.refresh` does only that much work.

**The fourth one is there because the third could not see the pass.** `analysis.py` has
always been in the `metrics` set, so an edit to it read as "the metrics changed" and the
repair was a re-score -- which reused the stored `analysis.json` and re-stamped the
measurement current with the pass never having run. `docs/archive/bugs-resolved.md` (the
`v5` entry) has the demonstration: one comment line, `stale (re-score)`, a re-score, `current`, and
`analysis.json` byte-identical throughout. The `metrics` fingerprint alone cannot say
whether the pass was among what moved, so the pass gets its own, and a basis stamped
before it had one is read at the coarser grain: its `metrics` still covers the pass, so
an unmoved `metrics` clears it, and a moved one takes the stronger repair.

**`tables` is per-ruleset and the other two are not, and that asymmetry is the point.**
It fingerprints what `engine/compile.py` and `engine/bridge.py` *produced* for one game
rather than their source, so an edit to the compiler that changes no ruleset's tables
leaves every stored measurement current. The old shape charged a full corpus replay for
any edit anywhere under `engine/`, and schema 6 paid one to reproduce numbers its own
inertness evidence had already shown were correct. That is a measured cost, not a
predicted one, and it is the one this split removes.

The split is about *correctness* first: games that are still valid should not be thrown
away. It is a speed win only where the games are expensive, because a re-score still pays
the bootstrap, which dominates scoring -- 41.8s against a 6.9s tournament on `antichess`.
See `corpus.refresh` for the measurement.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import tokenize
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

# The repository root, from this file's own location. Everything fingerprinted lives
# under it. An installed-but-not-editable copy has no `rust/src/` beside it, which is
# handled where it is read rather than by pretending the directory is there.
ROOT = Path(__file__).resolve().parent.parent

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
_MASK = (1 << 64) - 1


def _fnv1a(chunks: list[bytes]) -> str:
    """FNV-1a over a sequence of byte strings, as sixteen hex digits.

    Must stay byte-identical to `absorb` in `rust/build.rs`: the whole point of the
    build fingerprint is that the two sides compute the same number from the same files.
    A real digest would be no better here -- the question is only ever "same or not" --
    and it would cost the core a build-dependency.

    The arithmetic is `absorb`'s: xor the byte in, multiply by the prime, keep the low 64
    bits. What differs is how often the low 64 bits are kept. Reduction mod 2^64 commutes
    with both steps -- a xor with a byte touches only the low eight bits, and a product's
    low 64 bits depend only on its factors' low 64 bits -- so four bytes run unreduced and
    the mask lands once per block, with the same residue by construction. In pure Python
    the mask is the cost worth removing: it is an arbitrary-precision allocation per byte,
    where hoisting the constants alone buys a tenth and unrolling with a mask per byte
    buys nothing; blocks past six bytes lose the gain again to bigger integers. A
    256-entry table of the per-byte step does not exist: xor and multiplication do not
    commute, so the step is a function of all 64 bits of the state and not of the byte.
    Item 16 of `docs/performance-opportunities.md` has the numbers, and
    `tests/test_measurement.py` holds `absorb`'s own loop as the reference and pins this
    against it at every block phase.
    """
    data = b"".join(chunks)
    prime = FNV_PRIME
    mask = _MASK
    digest = FNV_OFFSET
    # Four strided slices zip into consecutive four-byte blocks: exactly len // 4 of them,
    # because the slices are unequal on purpose and the shortest, the fourth, decides.
    # `strict` would raise on the tail this loop is meant to leave behind.
    for b0, b1, b2, b3 in zip(data[0::4], data[1::4], data[2::4], data[3::4], strict=False):
        digest = (
            (((((((digest ^ b0) * prime) ^ b1) * prime) ^ b2) * prime) ^ b3) * prime
        ) & mask
    for byte in data[len(data) - len(data) % 4:]:
        digest = ((digest ^ byte) * prime) & mask
    return f"{digest:016x}"


def _fold_crlf(body: bytes) -> bytes:
    """The LF reading of a file, so line endings cannot move a fingerprint.

    `core.autocrlf=true` on this machine and the committed blobs are LF, so a fresh
    `.claude/worktrees/` checkout holds CRLF where the main checkout holds LF for the same
    commit. Hashing raw bytes made identical code fingerprint differently in the two
    checkouts, and since the drift reached `rust/src/` it put the difference inside the
    *build* fingerprint, whose only repair is a replay. So a campaign run in a worktree
    could not be delivered as current at all -- `docs/archive/bugs-resolved.md` has the
    60-spec measurement.

    A lone `\\r` is left alone. Git's conversion only ever adds or removes the *pair*, so
    folding a bare CR would buy nothing against the drift this exists for, and it would
    collapse two files that genuinely differ -- a fingerprint that fails to notice a change
    is the one direction that is never safe. LF-only input comes back untouched, so no LF
    fingerprint moves.

    `fold_crlf` in `rust/build.rs` folds the same way on the same bytes. The fold lives
    here rather than in `_fnv1a` because only a file body can carry a line ending. The
    other caller hashes a `json.dumps` blob, which escapes a CR rather than emitting one,
    so `_fnv1a` stays the plain FNV-1a that `absorb` is.
    """
    return body.replace(b"\r\n", b"\n")


def _hash_files(paths: list[Path], reading: Callable[[bytes], bytes] = _fold_crlf) -> str:
    """Fingerprint a set of source files by name and content.

    Sorted by file name so the answer is a property of the sources and not of the order
    the filesystem offered them. Bodies are read as bytes and passed through `reading`,
    which by default folds them to LF, so two checkouts of one commit agree however git
    wrote their newlines. `rust_source_fingerprint` passes `_rust_reading`, which
    `build.rs` computes too.
    """
    chunks: list[bytes] = []
    for path in sorted(paths, key=lambda p: p.name):
        try:
            body = path.read_bytes()
        except OSError:
            continue
        chunks.extend([path.name.encode("utf-8"), b"\0", reading(body), b"\0"])
    return _fnv1a(chunks)


# The Rust token reading (Q630, `docs/plan-compute.md` §2.1). `rust/build.rs` computes the
# same bytes from the same files, and `rust/reading-corpus/` pins both against one list
# of expected readings, so the two sides cannot drift apart without a test failing.
_RUST_SPACE = frozenset(b" \t\n\r\x0b\x0c")
_TEST_MODULE = b"#[cfg(test)]"


class _Unterminated(ValueError):
    """A literal or block comment that runs off the end of the file."""


def _ident_byte(byte: int) -> bool:
    return (
        byte >= 0x80
        or byte == 0x5F
        or 0x30 <= byte <= 0x39
        or 0x41 <= byte <= 0x5A
        or 0x61 <= byte <= 0x7A
    )


def _utf8_width(lead: int) -> int:
    if lead >= 0xF0:
        return 4
    if lead >= 0xE0:
        return 3
    if lead >= 0xC0:
        return 2
    return 1


def _string_end(src: bytes, index: int) -> int:
    """The end of a `"`-delimited literal whose body starts at `index`, escapes honoured."""
    while index < len(src):
        byte = src[index]
        if byte == 0x5C:
            index += 2
        elif byte == 0x22:
            return index + 1
        else:
            index += 1
    raise _Unterminated


def _rust_token(src: bytes, index: int) -> tuple[str, int]:
    """The token at `index`, as (kind, end): `space`, `comment`, `literal` or `byte`.

    `literal` is every string, byte string, C string, raw string and char literal, kept
    whole so nothing inside one is read as a comment or as whitespace. A `'` that does not
    open a char literal is a lifetime or a label and is an ordinary byte. Everything else
    outside a literal is one `byte` at a time, which is all the reading needs: it keeps
    every such byte and only has to know where comments, whitespace and literals are.
    """
    n = len(src)
    byte = src[index]
    if byte in _RUST_SPACE:
        return "space", index + 1
    following = src[index + 1] if index + 1 < n else -1
    if byte == 0x2F and following == 0x2F:  # `//`, `///`, `//!`
        end = src.find(b"\n", index)
        return "comment", n if end < 0 else end
    if byte == 0x2F and following == 0x2A:  # `/*`, which nests
        depth, at = 1, index + 2
        while depth:
            if at >= n:
                raise _Unterminated
            if src.startswith(b"/*", at):
                depth, at = depth + 1, at + 2
            elif src.startswith(b"*/", at):
                depth, at = depth - 1, at + 2
            else:
                at += 1
        return "comment", at
    if byte == 0x22:
        return "literal", _string_end(src, index + 1)
    if index == 0 or not _ident_byte(src[index - 1]):
        # A raw string: `r`, `br` or `cr`, any number of `#`, then `"`.
        at = index + 1 if byte in (0x62, 0x63) and following == 0x72 else index
        if src[at] == 0x72:
            quote = at + 1
            while quote < n and src[quote] == 0x23:
                quote += 1
            if quote < n and src[quote] == 0x22:
                close = b'"' + b"#" * (quote - at - 1)
                end = src.find(close, quote + 1)
                if end < 0:
                    raise _Unterminated
                return "literal", end + len(close)
    if byte == 0x27 and following >= 0:
        if following == 0x5C:  # `'\''`, `'\\'`, `'\u{..}'`: the escaped byte is skipped
            end = src.find(b"'", index + 3)
            if end < 0:
                raise _Unterminated
            return "literal", end + 1
        quote_at = index + 1 + _utf8_width(following)
        if quote_at < n and src[quote_at] == 0x27:
            return "literal", quote_at + 1
    return "byte", index + 1


def _test_module_end(src: bytes, index: int) -> int | None:
    """Where `#[cfg(test)] mod name { ... }` ends, if that is what starts at `index`.

    `index` is just past the attribute. Only the exact spelling rustfmt writes is
    recognised, followed by whitespace, `mod`, a plain name and a brace; anything else
    (another attribute between, a comment, `mod name;`, a `cfg(test)` on a field or an
    arm) answers `None` and is read as code, which is the safe direction.
    """
    n = len(src)

    def spaces(at: int) -> int:
        while at < n and src[at] in _RUST_SPACE:
            at += 1
        return at

    at = spaces(index)
    if not src.startswith(b"mod", at):
        return None
    at += 3
    name = spaces(at)
    # A plain ASCII name: not a digit first, not `r#`, nothing past ASCII.
    if name == at or name >= n or not _ident_byte(src[name]) or src[name] >= 0x80:
        return None
    if 0x30 <= src[name] <= 0x39:
        return None
    at = name
    while at < n and src[at] < 0x80 and _ident_byte(src[at]):
        at += 1
    at = spaces(at)
    if at >= n or src[at] != 0x7B:
        return None
    depth = 0
    while at < n:
        kind, end = _rust_token(src, at)
        if kind == "byte" and src[at] == 0x7B:
            depth += 1
        elif kind == "byte" and src[at] == 0x7D:
            depth -= 1
            if depth == 0:
                return end
        at = end
    raise _Unterminated


def _rust_token_reading(src: bytes) -> bytes:
    """The LF source of a `.rs` file with everything that cannot reach the build removed.

    Exactly: every comment (line, doc and nested block) and every `#[cfg(test)] mod`
    goes; each run of whitespace, comments and test modules outside a literal becomes one
    space, none at either end; every other byte, and every literal byte for byte, stays.
    Raises `_Unterminated` when a literal or comment runs off the end.
    """
    out = bytearray()
    pending = False
    index = 0
    n = len(src)
    while index < n:
        if src.startswith(_TEST_MODULE, index):
            end = _test_module_end(src, index + len(_TEST_MODULE))
            if end is not None:
                index, pending = end, True
                continue
        kind, end = _rust_token(src, index)
        if kind in ("space", "comment"):
            pending = True
        else:
            if pending and out:
                out.append(0x20)
            pending = False
            out += src[index:end]
        index = end
    return bytes(out)


def _rust_reading(body: bytes) -> bytes:
    """What `rust_source_fingerprint` hashes for one file: the token reading of its LF form.

    Rust is whitespace-insensitive outside literals, doc comments become `#[doc]`
    attributes nothing here reads, and `rust/src/` uses none of `line!`, `file!`,
    `column!`, `#[track_caller]` or `include_str!`, so a comment, a reflow or a test-only
    edit cannot change what the release build does, and none moves the fingerprint.
    Anything in doubt -- an unterminated literal or comment -- is read as its LF bytes,
    the more sensitive reading. `token_reading` in `rust/build.rs` is the same function.
    """
    body = _fold_crlf(body)
    try:
        return _rust_token_reading(body)
    except _Unterminated:
        return body


# A PEP 263 declaration. A file carrying one is hashed as bytes, because its comment
# decides how the rest of the file decodes.
_CODING = re.compile(rb"^[ \t\f]*#.*?coding[:=]", re.MULTILINE)


def _code_reading(body: bytes) -> bytes:
    """A Python source with its comments and docstrings taken out.

    Rewording prose must not stale a stored score, and changing code must. So the reading
    removes exactly what cannot reach a result:

    * every comment token;
    * every docstring, the first string statement of a module, class or function, which
      only sets `__doc__` (`tests/test_measurement.py` pins that no fingerprinted source
      reads one);
    * trailing whitespace and blank lines, except on lines a string literal spans, where
      both are part of the string's value.

    Everything else is kept as source text, byte for byte. The reading uses only node and
    token *positions*, never `ast.dump` or `ast.unparse`, whose output changes between
    Python versions. So the fingerprint is the same on every supported interpreter.

    Any doubt falls back to the LF bytes, which is the more sensitive reading: a file that
    does not parse, a lone CR, a coding declaration, or anything that is not UTF-8.
    """
    body = _fold_crlf(body)
    if b"\r" in body or _CODING.search(body, 0, _second_line_end(body)):
        return body
    try:
        text = body.decode("utf-8")
        tree = ast.parse(text)
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, ValueError, tokenize.TokenError):
        return body

    lines = text.split("\n")
    encoded = [line.encode("utf-8") for line in lines]

    def column(row: int, offset: int) -> int:
        # `ast` counts UTF-8 bytes along a line and `tokenize` counts characters.
        return len(encoded[row - 1][:offset].decode("utf-8"))

    # (row, column) spans in characters, end exclusive.
    removed: list[tuple[int, int, int, int]] = []
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
            and first.end_col_offset is not None
        ):
            docstrings.add(id(first.value))
            removed.append((
                first.lineno,
                column(first.lineno, first.col_offset),
                first.end_lineno,
                column(first.end_lineno, first.end_col_offset),
            ))
    for token in tokens:
        if token.type == tokenize.COMMENT:
            removed.append((*token.start, *token.end))

    # Rows whose line ends inside a string literal. Their whitespace and emptiness are
    # the string's content. Every string node counts, however nested, so a position an
    # older interpreter reports loosely can only protect a line, never expose one.
    strings = tuple(
        kind
        for kind in (ast.JoinedStr, getattr(ast, "TemplateStr", None))
        if isinstance(kind, type)
    )
    inside: set[int] = set()
    for node in ast.walk(tree):
        literal = isinstance(node, strings) or (
            isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))
        )
        first_row = getattr(node, "lineno", None)
        last_row = getattr(node, "end_lineno", None)
        if literal and id(node) not in docstrings and first_row and last_row:
            inside.update(range(first_row, last_row))

    # Cut from the last span backwards, so earlier coordinates stay valid.
    for start_row, start_col, end_row, end_col in sorted(removed, reverse=True):
        head = lines[start_row - 1][:start_col]
        tail = lines[end_row - 1][end_col:]
        lines[start_row - 1 : end_row] = [head + tail] + [""] * (end_row - start_row)

    kept: list[str] = []
    for row, line in enumerate(lines, start=1):
        if row in inside:
            kept.append(line)
        elif line.strip():
            kept.append(line.rstrip())
    return "\n".join(kept).encode("utf-8")


def _second_line_end(body: bytes) -> int:
    first = body.find(b"\n")
    if first < 0:
        return len(body)
    second = body.find(b"\n", first + 1)
    return len(body) if second < 0 else second



def _first(agents: list[str], shown: int = 2) -> str:
    """Name a couple of agents, and say how many more there are.

    A roster can differ in every rung, and a reason the length of the roster is a reason
    nobody reads. Two names and a count says the same thing.
    """
    if not agents:
        return "nothing"
    head = ", ".join(agents[:shown])
    rest = len(agents) - shown
    return f"{head} and {rest} more" if rest > 0 else head


def _listed(seeds: tuple[int, ...]) -> str:
    """`seed 2000`, or `seeds 1000, 2000 and 3000`."""
    if len(seeds) == 1:
        return f"seed {seeds[0]}"
    head = ", ".join(str(seed) for seed in seeds[:-1])
    return f"seeds {head} and {seeds[-1]}"


def _seeds(data: dict[str, Any]) -> tuple[int, ...]:
    """The seeds a stored basis names, sorted, however it spelled them.

    Two spellings, one meaning. `seeds` is a list; `seed` is what every basis written
    before a game could have more than one run recorded, and one seed is one run.
    Neither, or an empty list, is no basis at all -- raising here is how `from_dict`
    reports that.
    """
    found = data.get("seeds")
    if found is None:
        return (int(data["seed"]),)
    if isinstance(found, (str, bytes, int, float)):
        raise TypeError(f"seeds is not a list of seeds: {found!r}")
    answer = tuple(sorted(int(seed) for seed in found))
    if not answer:
        raise ValueError("a basis names at least one seed")
    return answer


def _sources(*patterns: str) -> list[Path]:
    return [path for pattern in patterns for path in ROOT.glob(pattern)]


# What decides the games, *once a ruleset's tables and its players exist*. A change to
# any of it means the tournament played is not the tournament that would be played now,
# so every downstream number is about a different match.
#
# `tournament/runner.py` fixes every game's seed, and `tournament/play.py` hands the core
# its piece values and writes each record. Either one changed means different games.
#
# **`agents/*.py` is deliberately in no set, and used to be in this one** (Q631,
# `docs/plan-compute.md` §2.3). The agents' Python decides a core-played game only through
# what `runner._run_native` hands `core.run_tournament`: one native spec per seat and the
# piece values. `Basis.players` hashes exactly that, per ruleset, the move `Basis.tables`
# made for the compiler. So an edit to `agents/nets.py`, `heuristics.py` or the fit-making
# half of `fitted.py` that hands the core nothing new stales nothing, and one that does
# stales every ruleset it reaches. Where the handoff cannot be read (a seat with no native
# form, a store this checkout lacks), `Basis.agents_source` -- the code reading of
# `AGENTS_SOURCES` -- decides instead, which is the old rule.
#
# **The compiler is deliberately in no set, and used to be in this one.** What a compiler
# produces is a better thing to fingerprint than a compiler is: `Basis.tables` hashes the
# arrays `bridge.build_tables` hands the core for *this* ruleset. The old set meant any
# edit anywhere under `engine/` invalidated every stored
# measurement in `runs/`, which schema 6 paid in full -- forty-two games replayed and
# rescored to reproduce numbers its own inertness evidence had already said were correct,
# because `rules.obstacles` has no entry in `build_tables` at all and no game before it
# played differently. The architecture claims a game is a function of the flattened
# tables plus the code that walks them; this takes the claim literally.
ENGINE_SOURCES = (
    "chess_variant_sandbox/tournament/play.py",
    "chess_variant_sandbox/tournament/runner.py",
)

# The players' source, read only where `Basis.players` cannot say: the fallback, never
# compared beside a players digest. Its own kind so it is never mistaken for `engine`.
AGENTS_SOURCES = ("chess_variant_sandbox/agents/*.py",)

# What turns finished games into a score. These read persisted records and never an
# engine, which is why a change here does not invalidate the games: they stand, and only
# the scoring pass is redone. `tournament/ratings.py` belongs here and not above -- the
# ladder is fitted from records after the fact, so a change to the fit re-scores without
# re-playing.
#
# `spec/complexity.py` belongs here too. It reads only the spec, but `rule_yield` divides
# by the `rule_mass` it computes, so a charge moved there moves a stored score. So does
# `spec/reachability.py`, because `piece_utilisation` counts only the types a player can
# hold. And every metric reads a game through `tournament/record.py`.
METRICS_SOURCES = (
    "chess_variant_sandbox/metrics/*.py",
    "chess_variant_sandbox/spec/complexity.py",
    "chess_variant_sandbox/spec/reachability.py",
    "chess_variant_sandbox/tournament/analysis.py",
    "chess_variant_sandbox/tournament/heuristics.py",
    "chess_variant_sandbox/tournament/ratings.py",
    "chess_variant_sandbox/tournament/record.py",
    "chess_variant_sandbox/engine/__init__.py",
    "chess_variant_sandbox/engine/engine.py",
    "chess_variant_sandbox/engine/moves.py",
    "chess_variant_sandbox/engine/state.py",
    "chess_variant_sandbox/agents/evaluation.py",
)

# The analysis pass: what turns finished games into `analysis.json`, which the scorer
# then reads as one more persisted record. It drives the engine, so it is not a metric,
# and it reads finished games, so it is not the engine either; a change to it leaves the
# games valid and the stored `analysis.json` stale, which is a repair of its own --
# `reanalyse` -- cheaper than a replay and more than a re-score.
#
# **It stays in `METRICS_SOURCES` as well, deliberately.** Every basis stamped before
# this set existed recorded a `metrics` fingerprint taken over a set that held `analysis.py`,
# and that number is what lets such a basis still be read: its `metrics` covers the pass,
# so where it matches, the pass has not moved. Taking the file out of `METRICS_SOURCES`
# would move the current `metrics` fingerprint for every stored measurement at once and
# report the whole corpus stale for an edit nobody made. The two sets overlapping on this
# one file is the point, and `tests/test_measurement.py` pins the overlap.
#
# The pass reads positions through the Python side of the engine boundary. `Engine` packs
# and unpacks each state, `moves.py` decodes each move, and `state.py` holds the result.
# None of the four decides a game, because the tournament runs whole inside the core over
# tables `Basis.tables` already hashes. So they charge a re-analysis, and they sit in
# `METRICS_SOURCES` too, as `analysis.py` does.
#
# `agents/evaluation.py` is the probe's ruler (`Evaluator.for_engine`, `cheapest_piece`,
# the goal and immunity weights `_score_roots` passes `root_scores`). It was covered only
# while `agents/*.py` sat in `engine`; since Q631 it is covered here, and in
# `METRICS_SOURCES` beside the rest of the pass. The blunder trials need nothing more:
# `finish_game` hands the core the roster's own native specs and values, which is
# `Basis.players`.
ANALYSIS_SOURCES = (
    "chess_variant_sandbox/tournament/analysis.py",
    "chess_variant_sandbox/engine/__init__.py",
    "chess_variant_sandbox/engine/engine.py",
    "chess_variant_sandbox/engine/moves.py",
    "chess_variant_sandbox/engine/state.py",
    "chess_variant_sandbox/agents/evaluation.py",
)

# Display prose that sits beside the metrics and reaches no score. The glossary computes
# nothing and only the page and the tests import it, so rewording an entry stales nothing.
# Left out of the code reading only; `tests/test_measurement.py` pins that no
# fingerprinted source imports it.
DISPLAY_ONLY = ("chess_variant_sandbox/metrics/glossary.py",)

_SOURCE_SETS = {
    "engine": ENGINE_SOURCES,
    "metrics": METRICS_SOURCES,
    "analysis": ANALYSIS_SOURCES,
    "agents": AGENTS_SOURCES,
}


def fingerprinted(kind: str) -> list[Path]:
    """The files one side's fingerprint reads."""
    prose = set(_sources(*DISPLAY_ONLY))
    return [path for path in _sources(*_SOURCE_SETS[kind]) if path not in prose]


@lru_cache(maxsize=len(_SOURCE_SETS))
def source_fingerprint(kind: str) -> str:
    """The current fingerprint of one side of the code, over its code reading.

    Cached: the tree does not move under a running process, and `discover` asks once per
    game.
    """
    return _hash_files(fingerprinted(kind), _code_reading)


@lru_cache(maxsize=16)
def code_fingerprint(*patterns: str) -> str:
    """The code reading of any files, for a stamp beside a score rather than of one.

    The same reading `source_fingerprint` takes, over globs a caller names. Nothing here
    moves a score's basis: the premise check (`premise.py`, Q502d) stamps its own code
    with it, and those files are in no set above.
    """
    return _hash_files(_sources(*patterns), _code_reading)


@lru_cache(maxsize=len(_SOURCE_SETS))
def bytes_fingerprint(kind: str) -> str:
    """One side read as LF bytes over the whole glob, glossary included.

    Every basis stamped before the code reading recorded this. `stale_against` still
    clears such a stamp where it matches, because unchanged bytes are unchanged code, so
    changing the reading moved no stored measurement. A re-score or replay re-stamps it
    under the code reading.
    """
    return _hash_files(_sources(*_SOURCE_SETS[kind]))


@lru_cache(maxsize=1)
def built_fingerprint() -> str:
    """What `rust/src/` hashed to when the loaded extension was compiled."""
    try:
        import chess_variant_sandbox._core as _core
    except ImportError:
        # The engine raises its own, far better, error the moment anything tries to
        # play. Reporting "unbuilt" as a fingerprint here keeps a *reader* -- a
        # leaderboard, a library listing -- from dying on an import it never needed.
        return "unbuilt"

    return str(getattr(_core, "BUILD_FINGERPRINT", "unknown"))


@lru_cache(maxsize=1)
def rust_source_fingerprint() -> str | None:
    """What `rust/src/` hashes to now, or ``None`` when the sources are not here.

    Absent for an installed copy without the repository beside it. That is not an error
    and must not be reported as staleness: it means the build cannot be checked, so the
    check is skipped rather than failed.
    """
    src = ROOT / "rust" / "src"
    if not src.is_dir():
        return None
    return _hash_files(list(src.glob("*.rs")), _rust_reading)


_TABLES: dict[str, str] = {}

# Where compiled answers outlive the process that compiled them (R9 of
# `docs/reference/perf-survey-2026-09.md`). Under `runs/`, so it is generated and
# gitignored like everything else there, and one file rather than a directory, because
# `runs/` is walked for directories that are runs. The variable moves it, and is read at
# every use rather than once, so a test's setting reaches a spawned worker too.
TABLES_CACHE = ROOT / "runs" / "tables-fingerprints.json"
TABLES_CACHE_ENV = "CHESS_VARIANT_SANDBOX_TABLES_CACHE"


def tables_cache_path() -> Path:
    """Where the on-disk tables cache is, `TABLES_CACHE` unless the variable says."""
    import os

    moved = os.environ.get(TABLES_CACHE_ENV)
    return Path(moved) if moved else TABLES_CACHE

# Moves when the cache file's own shape does, or when what a stored answer means does.
_TABLES_CACHE_FORMAT = 1

# Where the compile starts. Everything these import inside the package is followed.
_COMPILE_ENTRIES = (
    "chess_variant_sandbox/engine/compile.py",
    "chess_variant_sandbox/engine/bridge.py",
)

# `import chess_variant_sandbox.x.y` and `from chess_variant_sandbox.x import a, b`,
# anywhere in a file, deferred imports included. The names after `import` are kept so a
# submodule imported by name is found too.
_PACKAGE_IMPORT = re.compile(
    rb"(?:^|\s)(?:from\s+(chess_variant_sandbox(?:\.\w+)*)\s+import\s+\(?([\w\s,]+)"
    rb"|import\s+(chess_variant_sandbox(?:\.\w+)*))"
)


def _module_file(dotted: str) -> Path | None:
    """The source file a dotted package name refers to, or ``None`` for an attribute."""
    base = ROOT.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def compile_sources() -> list[Path]:
    """Every source file whose content can decide what a ruleset compiles to.

    Computed, never listed by hand: a hand list is how a stale fingerprint would get in
    silently, the first time somebody moved a helper the compiler reads into a new file.
    It is all of `engine/` and `spec/`, plus whatever the compiler's entry points import
    from elsewhere in the package, followed to closure, plus this file, which decides how
    the tables are turned into a number. More files than the compile reads only costs a
    recompile after an unrelated edit; fewer would serve a stale answer.
    """
    found = {
        path.resolve()
        for path in _sources(
            "chess_variant_sandbox/engine/**/*.py", "chess_variant_sandbox/spec/**/*.py"
        )
    }
    found.add((ROOT / "chess_variant_sandbox" / "measurement.py").resolve())
    return sorted(found | _closure(_COMPILE_ENTRIES))


def _closure(entries: tuple[str, ...]) -> set[Path]:
    """Every package file ``entries`` import, followed to closure, the entries included."""
    found: set[Path] = set()
    queue = [(ROOT / entry).resolve() for entry in entries]
    seen: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        found.add(path)
        try:
            body = path.read_bytes()
        except OSError:
            continue
        for match in _PACKAGE_IMPORT.finditer(body):
            module = (match.group(1) or match.group(3)).decode("ascii")
            names = [
                name.strip()
                for name in (match.group(2) or b"").decode("ascii").split(",")
                if name.strip()
            ]
            parts = module.split(".")
            # Importing `a.b.c` runs `a/__init__.py` and `a/b/__init__.py` first.
            parents = [".".join(parts[:end]) for end in range(1, len(parts))]
            named = [f"{module}.{name.split()[0]}" for name in names]
            for dotted in [*parents, module, *named]:
                target = _module_file(dotted)
                if target is not None:
                    queue.append(target.resolve())
    return found


@lru_cache(maxsize=1)
def compile_source_key() -> str:
    """A digest of every file in `compile_sources`, by path and exact bytes.

    Exact bytes, not the code reading `source_fingerprint` takes: this keys a cache, so
    a reworded comment costing one recompile is the right trade against parsing every
    source on every start. SHA-256 rather than `_fnv1a`, because nothing on the Rust side
    has to agree with it and the pure-Python FNV over the engine's sources would cost
    more than the cache saves. Cached for the process: the tree does not move under it.
    """
    return _source_key(f"tables-cache {_TABLES_CACHE_FORMAT}", compile_sources())


def _source_key(label: str, paths: list[Path]) -> str:
    """SHA-256 over a label and every file in ``paths``, by path and exact bytes."""
    import hashlib

    digest = hashlib.sha256(f"{label}\0".encode())
    for path in paths:
        try:
            body = path.read_bytes()
        except OSError:
            body = b"<unreadable>"
        name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        digest.update(name.encode("utf-8") + b"\0" + body + b"\0")
    return digest.hexdigest()


# Where `eval_values` starts, for the values cache's key. `play.py` imports the agents
# package, so the closure holds every agent module: an edit to any of them costs one
# recompile per ruleset, never a stale answer.
_VALUES_ENTRIES = (
    "chess_variant_sandbox/agents/evaluation.py",
    "chess_variant_sandbox/tournament/play.py",
)


def values_sources() -> list[Path]:
    """Every source file whose content can decide a ruleset's `eval_values`."""
    return sorted(set(compile_sources()) | _closure(_VALUES_ENTRIES))


@lru_cache(maxsize=1)
def values_source_key() -> str:
    """`compile_source_key`'s counterpart for the values cache, over `values_sources`."""
    return _source_key(f"values-cache {_TABLES_CACHE_FORMAT}", values_sources())


def values_cache_path() -> Path:
    """Beside the tables cache, so the one variable moves both."""
    return tables_cache_path().with_name("values-fingerprints.json")


# The on-disk cache as read by this process: which source key it was read under, and
# what it held. Read once; a miss adds to it and writes the whole file back.
_DISK: dict[str, Any] = {"path": None, "key": None, "tables": {}}
_VALUES_DISK: dict[str, Any] = {"path": None, "key": None, "tables": {}}


def _disk_tables() -> dict[str, str]:
    """The stored tables answers for the current sources."""
    return _read_disk(_DISK, tables_cache_path(), compile_source_key())


def _read_disk(state: dict[str, Any], path: Path, key: str) -> dict[str, str]:
    """The stored answers for the current sources, read at most once per process.

    An answer stored under any other source key is dropped, never consulted: the file
    holds one key's answers at a time, and a key that moved means every one of them may
    be wrong.
    """
    if state["path"] == path and state["key"] == key:
        return dict(state["tables"])
    tables: dict[str, str] = {}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = None
    if isinstance(stored, dict) and stored.get("sources") == key:
        found = stored.get("tables")
        if isinstance(found, dict):
            tables = {str(k): str(v) for k, v in found.items()}
    state.update(path=path, key=key, tables=tables)
    return dict(tables)


def _remember_tables(digest: str, answer: str) -> None:
    """Add one tables answer to the on-disk cache."""
    _write_disk(_DISK, tables_cache_path(), compile_source_key(), digest, answer)


def _write_disk(state: dict[str, Any], path: Path, key: str, digest: str, answer: str) -> None:
    """Add one answer to an on-disk cache, best effort.

    Written whole to a temporary file and renamed over the old one, so a reader never sees
    half a file. Two processes missing at once can each write, and the later rename
    drops the earlier one's addition; that costs a recompile later, never a wrong answer,
    since every answer in either file is the compile's own. A `runs/` that cannot be
    written (a read-only install) leaves the cache in memory only.
    """
    import os

    tables = _read_disk(state, path, key)
    tables[digest] = answer
    # Pick up what another process stored meanwhile, under the same key.
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(stored, dict) and stored.get("sources") == state["key"]:
            tables = {**{str(k): str(v) for k, v in stored.get("tables", {}).items()},
                      **tables}
    except (OSError, ValueError, AttributeError):
        pass
    state["tables"] = tables
    payload = json.dumps(
        {"sources": state["key"], "tables": dict(sorted(tables.items()))},
        indent=0, separators=(",", ":"),
    )
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()


def tables_fingerprint(spec: dict[str, Any]) -> str:
    """What this ruleset compiles to, as sixteen hex digits.

    The per-ruleset half of what decides a game. `engine/compile.py` resolves the whole
    rule DSL into per-(player, type, square) tables and `engine/bridge.py` flattens them
    into integer arrays; the core walks those arrays and knows nothing else about the
    rules. So two runs of the same ruleset are the same match exactly when these arrays
    and the code that walks them are the same -- which is what this measures, in place of
    measuring the source of the compiler that produced them.

    Keys sitting at an inert default are dropped first; see `bridge.canonical_tables` for
    why that is load-bearing rather than tidy.

    **Cached on the spec's mechanics hash**, which is exact rather than approximate: the
    hash covers `canonical.MECHANIC_KEYS` and the tables are built from nothing else, so
    two specs with the same hash compile to the same tables by construction. It is worth
    caching because compiling is not free -- the whole committed corpus is about three
    seconds, most of it in the wrapped boards -- and every reader that renders a score
    asks this.

    **And on disk, in `tables_cache_path()`**, keyed by the mechanics hash under
    `compile_source_key`, so `measure --check` and a cold listing compile nothing when
    nothing changed. The key covers every source the compile can read, and an answer
    stored under another key is never read, so an edit to the compiler costs a recompile
    and cannot serve a stale fingerprint.
    """
    from chess_variant_sandbox.spec.canonical import spec_hash

    digest = spec_hash(spec)
    found = _TABLES.get(digest)
    if found is not None:
        return found
    found = _disk_tables().get(digest)
    if found is not None:
        _TABLES[digest] = found
        return found

    from chess_variant_sandbox.engine.bridge import canonical_tables

    tables = canonical_tables(_compiled(spec, digest))
    # `sort_keys` so the answer is a property of the tables and not of the order the
    # builder happened to write them in, and `bytes` spelled out as hex because the
    # opening array and the unmoved mask cross as bytes and JSON has no word for them.
    blob = json.dumps(
        tables,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.hex() if isinstance(value, bytes) else repr(value),
    )
    answer = _fnv1a([blob.encode("utf-8")])
    _TABLES[digest] = answer
    _remember_tables(digest, answer)
    return answer


# The last ruleset compiled here, so `players_fingerprint` asking right after
# `tables_fingerprint` -- which is what `corpus.current_basis` does -- compiles once.
_LAST_COMPILED: dict[str, Any] = {}


def _compiled(spec: dict[str, Any], digest: str) -> Any:
    from chess_variant_sandbox.engine.compile import compile_spec

    if _LAST_COMPILED.get("digest") != digest:
        _LAST_COMPILED.clear()
        _LAST_COMPILED.update(digest=digest, spec=compile_spec(spec))
    return _LAST_COMPILED["spec"]


_VALUES: dict[str, str] = {}


def values_fingerprint(spec: dict[str, Any]) -> str:
    """The piece values `play.eval_values` hands the core for this ruleset, as a digest.

    Cached as `tables_fingerprint` is, on the mechanics hash in memory and on disk under
    `values_source_key`, so a check that finds nothing changed compiles nothing.
    """
    import hashlib

    from chess_variant_sandbox.spec.canonical import spec_hash

    digest = spec_hash(spec)
    found = _VALUES.get(digest)
    if found is not None:
        return found
    path, key = values_cache_path(), values_source_key()
    found = _read_disk(_VALUES_DISK, path, key).get(digest)
    if found is not None:
        _VALUES[digest] = found
        return found

    from chess_variant_sandbox.tournament.play import eval_values

    values = eval_values(_compiled(spec, digest))
    answer = hashlib.sha256(json.dumps(list(values)).encode("utf-8")).hexdigest()[:16]
    _VALUES[digest] = answer
    _write_disk(_VALUES_DISK, path, key, digest, answer)
    return answer


def _seat(native: tuple[Any, ...]) -> list[Any]:
    """One native spec as JSON: tuples as lists, bytes (a net) as their SHA-256."""
    import hashlib

    out: list[Any] = []
    for slot in native:
        if isinstance(slot, bytes):
            out.append("sha256:" + hashlib.sha256(slot).hexdigest())
        elif isinstance(slot, tuple):
            out.append(list(slot))
        else:
            out.append(slot)
    return out


def players_fingerprint(spec: dict[str, Any], agents: tuple[str, ...]) -> str:
    """What the runner hands the core for this roster on this ruleset, as a digest.

    `runner._run_native` hands `core.run_tournament` one native spec per seat
    (`agents.native_spec`) and the piece values (`play.eval_values`), and with the tables
    and the build that is everything an all-native game is a function of. So this hashes
    them, as `tables_fingerprint` hashes what the compiler produced: an edit under
    `agents/` that changes no seat's spec and no value moves nothing, and one that does
    moves every ruleset it reaches (Q631, `docs/plan-compute.md` §2.3).

    The fit vector and the net bytes are inside the specs, so this overlaps `Basis.fit`
    and `Basis.net` on purpose: those say which repair, this says whether the games stand.

    `""` means *cannot say*: a seat with no native form (the Python loop plays it, so its
    source decides), or a seat that does not resolve here -- most often a fit store this
    checkout lacks, as on the server for a visitor's design. `stale_against` then reads
    `agents_source` instead, the rule before Q631. Any error resolving a seat lands here,
    which is safe: the fallback is exact, only coarser.
    """
    from chess_variant_sandbox.agents import native_spec
    from chess_variant_sandbox.spec.canonical import spec_hash

    digest = spec_hash(spec)
    try:
        natives = [native_spec(agent, digest) for agent in agents]
    except Exception:  # noqa: BLE001 - cannot say, and the fallback is exact
        return ""
    if any(native is None for native in natives):
        return ""
    import hashlib

    seats = [
        [agent, _seat(native)]
        for agent, native in zip(agents, natives, strict=True)
        if native is not None
    ]
    blob = json.dumps(
        {"seats": seats, "values": values_fingerprint(spec)}, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def stale_build() -> str | None:
    """The reason the loaded extension is behind `rust/src/`, if it is.

    The footgun `CLAUDE.md` sets in bold, turned into a sentence. Every measurement path
    asks this before spending minutes on a tournament, because a stale core does not fail
    -- it quietly answers with the previous build.
    """
    on_disk = rust_source_fingerprint()
    if on_disk is None:
        return None
    if on_disk == built_fingerprint():
        return None
    return (
        "the loaded core was built from different sources than rust/src/ "
        "-- rebuild with `uv sync --extra dev --reinstall-package chess-variant-sandbox`"
    )


# Who played a measurement's games, and how far the server checked them
# (`docs/plan-client-compute.md` §5). A browser-made result is provisional: it is judged
# against the server's basis with `played` set, and it never becomes an official score.
BROWSER = "browser"
"""`Basis.played` for games a visitor's browser played. `""` is the server."""

LEGAL = "legal"
"""`Basis.checked` once the server has walked every game for legal moves and outcomes."""

SAMPLE = "sample"
"""The kind of `Basis.checked` that says n games were replayed byte for byte."""


def sample_check(n: int) -> str:
    """`Basis.checked` for a result whose sample of n games replayed exactly.

    The n is in the stamp so that a later change to the sample size is visible to every
    reader of a number checked under the old one.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"a sample is a positive number of games, not {n!r}")
    return f"{SAMPLE}:{n}"


def read_check(checked: str) -> tuple[str, int] | None:
    """`(LEGAL, 0)`, `(SAMPLE, n)`, or ``None`` for anything this module does not write."""
    if checked == LEGAL:
        return LEGAL, 0
    kind, colon, count = checked.partition(":")
    if kind != SAMPLE or not colon or not count.isdigit() or count != str(int(count)):
        return None
    n = int(count)
    return (SAMPLE, n) if n >= 1 else None


def provenance_reads(played: str, checked: str) -> bool:
    """Whether `played` and `checked` are a pair this module writes.

    The server's own measurement carries neither. A browser's carries `played`, with
    nothing checked yet as it leaves the browser, and `LEGAL` or a sample once the server
    has looked.
    """
    if not played:
        return not checked
    return played == BROWSER and (not checked or read_check(checked) is not None)


# Which objective the roster's fitted rungs played (`docs/reference/refit-repair.md`,
# Q80c). A fit is stamped with the core that made it and nothing else, so on another core
# "does this verdict still hold?" is answered only by re-deriving it, which is the fourth
# repair, `refit`. The field records the *player*, not the core: a refused fit plays the
# built-in objective whatever its vector says, so a refused vector that drifts moves nothing.
BUILTIN_PLAYER = "builtin"
"""`Basis.fit` where the fit was refused or could not be made: the built-in objective."""

UNVERIFIED = "unverified"
"""`Basis.fit` taken now, where the store was neither made nor reproduced on this core."""


@dataclass(frozen=True)
class FitReading:
    """The ruleset's fit store as it stands, handed to `Basis.current` by `corpus`.

    ``player`` is the store's digest (`BUILTIN_PLAYER`, or `fit:` and twelve hex digits
    over the procedure and the vector), ``made_on`` the core that made it (``""`` when the
    store records none) and ``verified_on`` every core it is known to be the answer on --
    its maker and every core a check reproduced it on. Read from the store and its sidecar
    by `tournament.fitting.fit_reading`, which this module cannot import.
    """

    player: str
    made_on: str = ""
    verified_on: frozenset[str] = frozenset()


def _core_of(engine: str) -> str:
    """The build fingerprint out of an `engine` stamp, `<sources>-<build>`."""
    return engine.rpartition("-")[2]


# Which net each `net=` procedure in the roster played (Q502f). A `net=` name is a
# training procedure, the same on every board, and absent or refused it is the built-in
# objective move for move (`learned-evaluator.md` §1), so the name alone does not say
# which player sat down. The stamp does: per procedure, `net:` and the first twelve hex of
# the SHA-256 of the bytes the core was sent, or `builtin` where it was sent none. Taken
# from `nets.net_bytes`, the one resolution every `net=` player goes through, so the
# stamp and the player cannot disagree. Integer inference makes a net the same player on
# every machine, so unlike a fit there is nothing for another core to re-derive: the
# core's own changes are `engine`'s.
UNREADABLE_NET = "unreadable"
"""A `Basis.net` player whose store could not be read: no measurement could have used it."""


def _net_procedures(agents: tuple[str, ...]) -> list[str]:
    """Every `net=` procedure the roster names, normalised, sorted, once each."""
    named = [agent for agent in agents if "net=" in agent]
    if not named:
        return []
    from chess_variant_sandbox.agents.base import parse_agent
    from chess_variant_sandbox.agents.nets import normalise_procedure

    found = set()
    for agent in named:
        procedure = parse_agent(agent).options.get("net")
        if procedure is not None:
            found.add(normalise_procedure(procedure))
    return sorted(found)


def net_players(agents: tuple[str, ...], ruleset_hash: str) -> str:
    """`Basis.net` for this roster on this ruleset: `""` where no agent names a net.

    `<procedure>=net:<12 hex>` or `<procedure>=builtin` per procedure, comma-joined in
    procedure order. Read from disk (and from nets registered in this process) the way the
    players read it, so it says what a game played now would be played by.
    """
    procedures = _net_procedures(agents)
    if not procedures:
        return ""
    import hashlib

    from chess_variant_sandbox.agents.nets import net_bytes

    parts = []
    for procedure in procedures:
        try:
            blob = net_bytes({"net": procedure}, ruleset_hash)
        except (OSError, ValueError):
            player = UNREADABLE_NET
        else:
            player = "net:" + hashlib.sha256(blob).hexdigest()[:12] if blob else BUILTIN_PLAYER
        parts.append(f"{procedure}={player}")
    return ",".join(parts)


def _player(fit: str) -> str:
    """A `Basis.fit` value in words."""
    if fit == BUILTIN_PLAYER:
        return "the built-in objective"
    return f"the fitted objective {fit}"


@dataclass(frozen=True)
class Basis:
    """Everything a stored measurement depends on.

    Compared as a whole. Every field is here because changing it alone has, at some
    point, changed a published number without changing anything that was written down.
    """

    ruleset_hash: str
    weights_version: str
    agents: tuple[str, ...]
    games_per_pairing: int
    # Every seed this measurement covers, sorted. One stamp covers all of a game's runs,
    # because nothing that stales a run -- roster, core, tables, pass, metrics, weights --
    # depends on the seed, so they stale together or not at all. A file therefore never
    # describes a mixed set: `corpus._store` writes it once, after every run it names is
    # current under one basis.
    seeds: tuple[int, ...]
    engine: str
    metrics: str
    # What the compiler produced for *this* ruleset. Separate from `engine` because the
    # two invalidate for different reasons and one of them is per-game: a change to the
    # walker or the players is about every ruleset at once, and a change to a ruleset's
    # tables is about that ruleset alone. `""` is what a spec this checkout cannot
    # compile reports, and it compares unequal to every real fingerprint, which is the
    # safe direction.
    tables: str = ""
    # What produced `analysis.json`. `""` here means *not recorded* rather than "nothing":
    # every basis stamped before the pass had its own fingerprint reads so, and
    # `stale_against` falls back to `metrics` for it, which covers the same file at a
    # coarser grain. That is the one field allowed to be absent, and `from_dict` says why.
    analysis: str = ""
    # Which rule turned the runs' own verdicts into the game's. Stamped by value, as the
    # roster is, because `corpus.py` is in no fingerprint set -- so changing the answer is
    # a one-line edit that stales every multi-run file by name. Compared only when the
    # stored file names more than one seed: a vote of one is the run's own verdict under
    # every rule.
    verdict_rule: str = ""
    # Who played the games, and how far the server has checked them. Both are `""` on
    # every measurement the server played itself, which is every measurement in `runs/`
    # and every stamp written before a browser could measure, so neither is written for
    # one and no stored file changes shape. A visitor's browser stamps `played` as
    # `BROWSER` and leaves `checked` empty, because a browser cannot vouch for itself;
    # the server writes `LEGAL` once every game has walked legal, and `sample_check(n)`
    # once n of them have replayed byte for byte (`docs/plan-client-compute.md` §5).
    #
    # `played` is compared like any other field, so a browser's games are never current
    # against the server's own basis: an official reader shows no number for them, which
    # is what keeps a provisional result out of every sort, tag and canonical file even
    # if one were written where a canonical file lives. `checked` is not compared. It
    # says who may see the number and how it is labelled, and the number is the same
    # number on the same games at every level of checking.
    played: str = ""
    checked: str = ""
    # Which objective the fitted rungs played: `BUILTIN_PLAYER` or `fit:<12 hex>`, the
    # store's digest over the procedure and the vector (Q80c). Taken now it is `UNVERIFIED`
    # when the store was neither made nor reproduced on the loaded core, since only a refit
    # can say what this core would play. `""` means *not recorded*: every stamp written
    # before the field, and a basis taken now for a ruleset with no store. `stale_against`
    # reads an absent stored value through `fit_reading` (the rule `analysis` follows: a
    # covering fact decides, and nothing is rewritten), and skips the check when the
    # basis taken now was told nothing.
    fit: str = ""
    # Which net each `net=` procedure in the roster played: `net_players`. `""` where no
    # agent names a net, which is every canonical roster so far, so no stamp written before
    # the field changes shape or reads stale. Compared like the roster: a different net is
    # a different player, and the games are a replay.
    net: str = ""
    # What the runner hands the core for this roster on this ruleset: `players_fingerprint`
    # (Q631). Per-ruleset like `tables`, and for the same reason: an agents edit that hands
    # the core nothing new leaves the games standing. `""` on
    # either side means *cannot say*, and `agents_source` decides instead; a stamp from
    # before the field records neither, so it reads stale by replay.
    players: str = ""
    # The code reading of `AGENTS_SOURCES` when this was stamped. Compared only where the
    # players cannot be read now, so it is the old rule exactly where the new one is blind.
    agents_source: str = ""
    # Today's `bytes_fingerprint` for `engine`, `metrics` and `analysis`. Set only on a
    # basis taken now, never stored and never part of equality. A stamp from before the
    # code reading holds these values, and it clears where it still matches them.
    engine_bytes: str = field(default="", compare=False, repr=False)
    metrics_bytes: str = field(default="", compare=False, repr=False)
    analysis_bytes: str = field(default="", compare=False, repr=False)
    # The seed sets this project uses, handed over by `corpus.current_basis`. Passed in
    # rather than imported for the reason the roster is: which seeds a score means is a
    # corpus decision, and this module's job is to record the answer and compare it.
    # Empty means *not told*, and the check is then skipped rather than failed.
    known_seed_sets: tuple[tuple[int, ...], ...] = field(
        default=(), compare=False, repr=False
    )
    # The fit store as it stands, on a basis taken now; never stored. `None` means *not
    # told*, and the fit check is then skipped, as it is for `known_seed_sets`.
    fit_reading: FitReading | None = field(default=None, compare=False, repr=False)

    @classmethod
    def current(
        cls,
        ruleset_hash: str,
        weights_version: str,
        agents: tuple[str, ...],
        games_per_pairing: int,
        seeds: tuple[int, ...],
        tables: str = "",
        verdict_rule: str = "",
        known_seed_sets: tuple[tuple[int, ...], ...] = (),
        fit: FitReading | None = None,
        players: str = "",
    ) -> Basis:
        """The basis a measurement taken right now would carry.

        The roster is passed in rather than read from `corpus`, which imports this
        module. It is also the honest shape: *which* roster a score means is a corpus
        decision, and this module's job is only to record the answer and compare it.

        `tables` is passed in for the same reason and one more: computing it needs the
        spec, and this module is imported by readers that hold a hash and never a spec.
        `corpus.current_basis` is the one place that has both.

        `fit` is the ruleset's fit store, passed in for the roster's reason. The stamp is
        its player where the loaded core made or reproduced it, and `UNVERIFIED` where
        not; no store at all records nothing.

        `players` is `players_fingerprint` for this roster, passed in for the tables'
        reason; `""` says it cannot be read, and the agents' source decides instead.
        """
        built = built_fingerprint()
        if fit is None:
            player = ""
        elif built in fit.verified_on:
            player = fit.player
        else:
            player = UNVERIFIED
        return cls(
            ruleset_hash=ruleset_hash,
            weights_version=weights_version,
            agents=tuple(agents),
            games_per_pairing=games_per_pairing,
            seeds=tuple(sorted(seeds)),
            engine=source_fingerprint("engine") + "-" + built,
            metrics=source_fingerprint("metrics"),
            tables=tables,
            analysis=source_fingerprint("analysis"),
            verdict_rule=verdict_rule,
            fit=player,
            net=net_players(tuple(agents), ruleset_hash),
            players=players,
            agents_source=source_fingerprint("agents"),
            engine_bytes=bytes_fingerprint("engine") + "-" + built,
            metrics_bytes=bytes_fingerprint("metrics"),
            analysis_bytes=bytes_fingerprint("analysis"),
            known_seed_sets=known_seed_sets,
            fit_reading=fit,
        )

    def in_browser(self, checked: str = "") -> Basis:
        """This basis as a visitor's browser measured it, checked as far as `checked` says.

        The server builds the basis a provisional result is judged against this way: the
        same game, roster, seeds, weights and fingerprints it would stamp itself, with
        the one field that says someone else played the games.
        """
        return replace(self, played=BROWSER, checked=checked)

    def as_dict(self) -> dict[str, Any]:
        stamp: dict[str, Any] = {
            "ruleset_hash": self.ruleset_hash,
            "weights_version": self.weights_version,
            "agents": list(self.agents),
            "games_per_pairing": self.games_per_pairing,
            "seeds": list(self.seeds),
            "engine": self.engine,
            "metrics": self.metrics,
            "tables": self.tables,
            "analysis": self.analysis,
            "verdict_rule": self.verdict_rule,
        }
        # Only where a browser played the games, so every server-made file is written
        # exactly as it was before the fields existed.
        if self.played:
            stamp["played"] = self.played
            stamp["checked"] = self.checked
        # Only where it says something, so a stamp for a ruleset with no store is written
        # as it was before the field existed.
        if self.fit:
            stamp["fit"] = self.fit
        # Likewise: only a roster naming a net writes it.
        if self.net:
            stamp["net"] = self.net
        # Only where recorded, so a basis built by hand keeps the shape it always had.
        if self.players:
            stamp["players"] = self.players
        if self.agents_source:
            stamp["agents_source"] = self.agents_source
        return stamp

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Basis | None:
        """Rebuild a basis from what a run wrote, or ``None`` if it wrote none.

        ``None`` is the common case for anything measured before this existed, and it is
        reported as unknown rather than as current. Refusing to guess is the whole
        mechanism: a measurement that cannot say what it rests on is one that has to be
        taken again.
        """
        if not isinstance(data, dict):
            return None
        # Absent is the server, which is every stamp written before a browser could
        # measure. Present, the pair has to read as a pair this module writes, or the
        # stamp is not one anybody can match: a result that cannot say who played it or
        # how far it was checked is one whose audience and label are unknown.
        played = str(data.get("played") or "")
        checked = str(data.get("checked") or "")
        if not provenance_reads(played, checked):
            return None
        try:
            return cls(
                ruleset_hash=str(data["ruleset_hash"]),
                weights_version=str(data["weights_version"]),
                agents=tuple(str(agent) for agent in data["agents"]),
                games_per_pairing=int(data["games_per_pairing"]),
                # A basis written before a game could have more than one run recorded
                # `seed: 0`, and that is exactly `seeds: (0,)` -- one run, from seed 0.
                # Reading it that way is what lets the staleness name the true repair,
                # which is to re-score seed 0 and play whatever else is expected, rather
                # than a replay of everything.
                seeds=_seeds(data),
                engine=str(data["engine"]),
                metrics=str(data["metrics"]),
                # Required, like every other field. A basis written before the tables
                # were fingerprinted cannot say what its games were played over, and
                # `UNKNOWN` -- replay -- is the only honest reading of that. Defaulting
                # it to the current value would let exactly the measurements this was
                # built to distinguish pass as current.
                tables=str(data["tables"]),
                # The one field that may be absent, and the reason is not leniency: a
                # basis written before the pass had its own fingerprint still recorded
                # `metrics`, which was taken over `analysis.py` too. So the pass *is*
                # covered, at a coarser grain, and `stale_against` reads it there --
                # an unmoved `metrics` clears the pass, a moved one takes the stronger
                # repair because it cannot say which side moved. Treating absence as
                # stale would instead replay every measurement in `runs/` for a change
                # nobody made; treating it as current would be the `tables` mistake.
                analysis=str(data.get("analysis") or ""),
                # Absent on every basis stamped before the runs could disagree, where it
                # means *a vote of one*. `stale_against` compares it only where there is
                # more than one run, so absence never stales a single-run file.
                verdict_rule=str(data.get("verdict_rule") or ""),
                played=played,
                checked=checked,
                # Absent on every stamp written before Q80c, and read in `stale_against`
                # through the store's own record rather than as agreement.
                fit=str(data.get("fit") or ""),
                # Absent on every stamp whose roster named no net, where it means none.
                net=str(data.get("net") or ""),
                # Absent on every stamp before Q631, where it means *not recorded*.
                players=str(data.get("players") or ""),
                agents_source=str(data.get("agents_source") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def stale_against(self, current: Basis) -> Staleness:
        """Why this measurement is not the measurement that would be taken now.

        Two kinds of answer since a game can have several runs. The seeds the file does
        not name owe a *replay of those seeds alone*, and the seeds it does name owe
        whatever the rest of the basis says -- `missing_seeds` and `stored_needs` carry
        the two apart, and `needs` stays the strongest of them so every reader that only
        wants "how much work" reads it unchanged.
        """
        reasons: list[str] = []
        # What the runs already in the file owe. The seeds that are not in it owe a
        # replay of their own, tracked apart so `measure` can play three seeds and
        # re-score none.
        needs = ""
        missing = tuple(seed for seed in current.seeds if seed not in self.seeds)

        if self.ruleset_hash != current.ruleset_hash:
            # Not staleness so much as a category error: two hashes are two games. It is
            # reported here because a caller holding the wrong directory should be told
            # that, rather than shown a confident number about a different ruleset.
            reasons.append(
                f"measured a different ruleset ({self.ruleset_hash}, not "
                f"{current.ruleset_hash})"
            )
            needs = "replay"

        if self.played != current.played:
            # Not a change in any code: a different kind of measurement. A browser's
            # games checked by the server are still not games the server played, so an
            # official reader shows no number for them, and the repair is the server's
            # own measurement, which is a replay.
            reasons.append(
                "played in a visitor's browser, not by the server"
                if self.played
                else "played by the server, not in a visitor's browser"
            )
            needs = "replay"

        if self.agents != current.agents or self.games_per_pairing != current.games_per_pairing:
            # Shape first, and *which agents* when the shape matches. Since `v9` the
            # roster can differ with the same count and the same games a pairing --
            # every searching rung carries a fitted objective, so `minimax:nodes=6400`
            # and `minimax:fit=...,nodes=6400` are the same shape and different players.
            # Reporting only the shape rendered that as "different roster (8 agents at 4
            # games a pairing, not 8 at 4)", which reads as a bug rather than a reason.
            if len(self.agents) == len(current.agents) and (
                self.games_per_pairing == current.games_per_pairing
            ):
                gone = [a for a in self.agents if a not in current.agents]
                came = [a for a in current.agents if a not in self.agents]
                detail = f"{_first(gone)} gave way to {_first(came)}"
            else:
                detail = (
                    f"{len(self.agents)} agents at {self.games_per_pairing} games a "
                    f"pairing, not {len(current.agents)} at {current.games_per_pairing}"
                )
            reasons.append(f"played on a different roster ({detail})")
            needs = "replay"
        if missing:
            # Not the whole measurement: the runs the file does name still stand, and
            # `measure` plays only these. More seeds than expected is never stale -- a
            # retired game and a candidate tested before promotion keep the better
            # measurement they have. This one reason does not write `needs`, which is
            # what the stored runs owe; the two are joined at the end.
            reasons.append(f"not yet measured on {_listed(missing)}")
        if current.known_seed_sets and self.seeds not in current.known_seed_sets:
            reasons.append("measured on seeds the project does not use")
            needs = "replay"
        if _moved(self.engine, current.engine, current.engine_bytes):
            reasons.append("the core, the schedule or the game loop changed since it was played")
            needs = "replay"
        # The players (Q631). Where both sides read them, they decide; where either could
        # not, the agents' source does, which is the rule before the players were read. A
        # stamp from before Q631 records neither, and is a replay.
        if self.players and current.players:
            if self.players != current.players:
                reasons.append("the players the core is handed changed since it was played")
                needs = "replay"
        elif self.agents_source != current.agents_source:
            reasons.append(
                "the agents changed since it was played, and what they hand the core "
                "cannot be read on both sides"
                if self.agents_source
                else "it does not record what the core was handed to play"
            )
            needs = "replay"
        if self.tables != current.tables:
            # The per-ruleset half, and the one that is usually *not* the reason. An edit
            # to `engine/compile.py` reaches here only if it changed what this ruleset
            # compiles to; if it did not, this field is equal and the measurement stands.
            reasons.append("this ruleset compiles to different tables than it was played on")
            needs = "replay"

        # Which objective the fitted rungs played (Q80c). Its own block, after the replay
        # block: a changed player is a replay, and a fit this core has not re-derived is a
        # flag, `refit`, that becomes `needs` only when nothing else is owed.
        refit = False
        reading = current.fit_reading
        if reading is not None and current.fit:
            if current.fit == UNVERIFIED:
                refit = True
                reasons.append(
                    f"its fit was made on core {reading.made_on} and has not been "
                    "re-derived on this one"
                    if reading.made_on
                    else "its fit records no core and has not been re-derived on this one"
                )
            else:
                # Absent is every stamp written before the field. The store says which
                # objective played where the core that played the games made or
                # reproduced it; anywhere else nothing says, and those games were played
                # on another core, which the check above has already sent to a replay.
                played = self.fit or (
                    reading.player if _core_of(self.engine) in reading.verified_on else ""
                )
                if not played:
                    reasons.append(
                        "it does not record which objective played, and its fit was not "
                        "made or reproduced on the core that played it"
                    )
                    needs = "replay"
                elif played != current.fit:
                    reasons.append(
                        f"it played {_player(played)}, and {_player(current.fit)} plays now"
                    )
                    needs = "replay"

        # Which nets played (Q502f). The roster check above catches a changed name; this
        # catches the same name resolving to other weights, or to none.
        if self.net != current.net:
            reasons.append(
                f"its nets were {self.net or 'none'}, and {current.net or 'none'} play now"
            )
            needs = "replay"

        # Below here the games still stand. The checks run in descending order of repair
        # -- replay above, then re-analyse, then re-score -- and each writes `needs` only
        # when nothing stronger has: a replay implies a re-analysis implies a re-score,
        # and the stronger requirement must not be overwritten by the weaker one.
        metrics_moved = _moved(self.metrics, current.metrics, current.metrics_bytes)
        # A basis stamped before the pass had a fingerprint of its own reads `""` here,
        # and is judged on `metrics`, which was taken over the pass's source too: an
        # unmoved `metrics` clears the pass, and a moved one cannot say whether the pass
        # was among what moved, so the honest reading of "cannot say" is the stronger
        # repair.
        analysis_moved = (
            _moved(self.analysis, current.analysis, current.analysis_bytes)
            if self.analysis
            else metrics_moved
        )
        if analysis_moved:
            reasons.append(
                "the analysis pass changed since it was run"
                if self.analysis
                else "the metrics changed since it was scored, and it does not record "
                "whether the analysis pass was among them"
            )
            needs = needs or "reanalyse"

        if self.weights_version != current.weights_version:
            reasons.append(
                f"scored under weights {self.weights_version}, not "
                f"{current.weights_version}"
            )
            needs = needs or "rescore"
        if metrics_moved and not analysis_moved:
            # The pass is inside the metrics set, so when it moved the reason above has
            # already said so more precisely; this is the scorer, the ladder or the
            # heuristics moving on their own.
            reasons.append("the metrics changed since it was scored")
            needs = needs or "rescore"
        if len(self.seeds) > 1 and self.verdict_rule != current.verdict_rule:
            # Only where there was a vote. A game with one run has the run's own verdict
            # under every rule, so comparing the rule there would stale a visitor's score
            # for a decision that could not have touched it.
            reasons.append(
                f"its verdict was taken under {self.verdict_rule or 'no named rule'}, "
                f"not {current.verdict_rule}"
            )
            needs = needs or "rescore"
        # Last, so every stronger repair has had its say: a refit is owed alongside any of
        # them, and is the whole of the work only when nothing else is.
        if refit and not needs:
            needs = "refit"

        return Staleness(
            reasons=tuple(reasons),
            needs="replay" if missing else needs,
            missing_seeds=missing,
            stored_needs=needs,
            stored_runs=len(self.seeds),
            expected_runs=len(current.seeds),
            refit=refit,
        )


def _moved(stored: str, now: str, bytes_now: str) -> bool:
    """Whether a stored source fingerprint matches today's under neither reading.

    A stamp under the code reading matches `now`, and one from before it matches
    `bytes_now`. Accepting the second is as strict as the bytes reading always was.
    """
    return stored != now and not (bytes_now and stored == bytes_now)


@dataclass(frozen=True)
class Staleness:
    """Why a measurement is out of date, and how much work it takes to fix.

    ``needs`` is ``""`` when it is current, ``"rescore"`` when the games stand and only
    the scoring pass has to run again, ``"reanalyse"`` when the games stand but the
    analysis pass that wrote `analysis.json` has changed, so that file has to be
    rewritten before the scoring pass reads it, and ``"replay"`` when the tournament
    itself is about a match nobody would play now. Each implies the ones before it.
    ``"refit"`` is the fourth and weakest (Q80c): nothing else is owed, but the fit the
    roster plays was not re-derived on the loaded core, and `refit` says so whatever
    ``needs`` is.

    **The first of those only became reachable after an engine edit when `Basis` grew
    `tables`.** Before that, every change under `chess_variant_sandbox/engine/*.py` answered "replay"
    for every ruleset in the corpus, whether or not it changed what any of them compiles
    to -- so "nothing" was an answer the harness could give about a weights version and
    never about the engine. It is now the *expected* answer for a compiler change that
    moves no ruleset's tables, and `tests/test_measurement.py` pins it.
    """

    reasons: tuple[str, ...] = ()
    needs: str = ""
    # The seeds the reader expects that the file does not name. `measure` plays these and
    # nothing else, and they are the one repair that costs less than it looks: three
    # missing runs of four leave the fourth alone.
    missing_seeds: tuple[int, ...] = ()
    # What the runs already in the file owe, which is `needs` minus the missing seeds.
    stored_needs: str = ""
    stored_runs: int = 0
    expected_runs: int = 0
    # The fourth repair (Q80c). The fit this measurement's roster resolves to was neither
    # made nor reproduced on the loaded core, so only re-deriving it can say whether the
    # players are the ones that played. Owed beside whatever `needs` says, and `needs` is
    # `"refit"` only when nothing else is; `corpus.refresh` pays it first, then reads the
    # staleness again.
    refit: bool = False

    def __bool__(self) -> bool:
        return bool(self.needs)

    @property
    def current(self) -> bool:
        return not self.needs

    def summary(self) -> str:
        """One line a person can act on, or an empty string when there is nothing to say."""
        if not self.reasons:
            return ""
        return f"stale ({self.cost()}): " + "; ".join(self.reasons)

    def cost(self) -> str:
        """The repair in words, counting runs when the two halves differ.

        `re-score` where a game has one run and every run owes the same thing. Where some
        runs are missing and the rest are not, the count is the news: `replay 1 of 4 runs,
        re-score 3` says the work is one tournament and three scorings, which is a
        different afternoon from four tournaments.
        """
        if not self.missing_seeds:
            return REPAIR_NAMES.get(self.needs, "replay")
        parts = [f"replay {len(self.missing_seeds)} of {self.expected_runs} runs"]
        if self.stored_needs and self.stored_runs:
            parts.append(f"{REPAIR_NAMES[self.stored_needs]} {self.stored_runs}")
        return ", ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "needs": self.needs,
            "reasons": list(self.reasons),
            "summary": self.summary(),
            "missingSeeds": list(self.missing_seeds),
            "storedNeeds": self.stored_needs,
            "refit": self.refit,
        }


# How each repair is spelled for a person, weakest first: each implies none before it and
# the three after `refit` each imply the ones before them. `""` is current. `refit` is
# first because it is what is owed only when nothing else is (Q80c): it re-derives the fit
# and implies no replay unless the player changed.
REPAIR_NAMES = {
    "refit": "refit",
    "rescore": "re-score",
    "reanalyse": "re-analyse",
    "replay": "replay",
}

# The state a measurement is in when nothing recorded what it rested on. Distinct from
# every other kind of staleness: the others name a difference, this one names an absence,
# and the repair is the same either way.
UNKNOWN = Staleness(
    ("it does not record what it was measured under -- taken before this was tracked",),
    "replay",
)


@dataclass(frozen=True)
class Measurement:
    """One stored canonical-roster measurement, and whether it still holds.

    The unit the readers deal in. Score, ladder and staleness travel together and come
    from the same directory -- which is the bug this type retires: the library used to
    pick the score by newest weights version and the ladder by biggest roster, in two
    independent scans, and would happily render one game's fresh Elo beside another
    run's year-old verdict.
    """

    hash: str
    directory: Path
    stored: dict[str, Any]
    basis: Basis | None
    staleness: Staleness

    @property
    def current(self) -> bool:
        return self.staleness.current

    @property
    def standings(self) -> list[dict[str, Any]]:
        return list(self.stored.get("standings") or [])

    def as_dict(self) -> dict[str, Any]:
        """What a reader renders. The staleness is never optional and never omitted."""
        return {
            **{k: v for k, v in self.stored.items() if k != "standings"},
            "source": self.directory.name,
            "staleness": self.staleness.as_dict(),
        }

    def save(self, path: Path) -> None:
        payload = dict(self.stored)
        payload["basis"] = self.basis.as_dict() if self.basis else None
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


SINGLE_RUN_FORM = "stored in the single-run form"


def _at_least_rescore(needs: str) -> str:
    return "rescore" if needs in ("", "refit") else needs


def load(path: Path, current: Basis, old_path: bool = False) -> Measurement | None:
    """Read a stored measurement and judge it against the basis that holds now.

    ``old_path`` says the file was found where a measurement lived before a game could
    have more than one run, `runs/<hash>-<seed>/`. Such a file is always stale, and so is
    any file with no `runs` list wherever it sits -- the rule does not rest on the path
    alone, because a reader that trusted the path would trust a copy. Either way the
    repair is at least a re-score, which writes the game's file in the new shape.
    """
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(stored, dict):
        return None
    basis = Basis.from_dict(stored.get("basis"))
    staleness = UNKNOWN if basis is None else basis.stale_against(current)
    if old_path or not isinstance(stored.get("runs"), list):
        # A refit is weaker than the re-score this form owes, so it gives way to it.
        staleness = replace(
            staleness,
            reasons=(*staleness.reasons, SINGLE_RUN_FORM),
            needs=_at_least_rescore(staleness.needs),
            stored_needs=_at_least_rescore(staleness.stored_needs),
        )
    return Measurement(
        hash=str(stored.get("hash", "")),
        directory=path.parent,
        stored=stored,
        basis=basis,
        staleness=staleness,
    )
