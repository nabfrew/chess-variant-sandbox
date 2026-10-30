"""Does the translated variant offer the same moves?

`variant.py` is allowed to be optimistic. This is where that is paid for, for the moves.

**What this establishes is the same legal moves in every position it walks, not the same
result.** Nothing here asks either engine who has won. The walk stops at the first
position our engine calls over, so an ending is never put to the other engine. A rule
that only decides the result (`checkmate`, `stalemate`, `max_plies`, the draw clocks,
`extinction`, `reach`, `check_limit`) is checked by `results.check_results` instead,
which asks both engines who won at the same positions (Q404). The one exception here is
an immediate ending of theirs that ours lacks, which empties their move list where the
walk meets it. `docs/reference/fairy-stockfish.md` (Q96, Q404) names the keys.

Before any position is compared, a backend that can say which keys of the definition it
did not read (the binary, `UciEngine.unread_options`) is asked, and an unread key with no
reason in `unread.UNREAD` refuses the translation (Q915). That is the one result question
put to the binary: a goal it dropped in silence offered the same moves everywhere.

One difference in the move lists is accepted, and only one: a drop that `drops.no_mate`
forbids, which theirs lists as a move that loses and ours leaves out (`_tolerated`, Q453).
Their engine is asked whether the drop mates; this never decides it.

The check is the one this project already trusts for movegen and it is trusted here for
the same reason: **a perft that agrees is two independent implementations agreeing about
a tree, and a rule either of them has wrong moves it.** What is added on top is a random
walk, because perft from the opening array only ever exercises the opening -- a
promotion rule, a drop region or a check counter is fifteen plies away and a shallow
perft never meets one.

Three failures are distinguished, and the distinction is what makes a report worth
reading:

* **refused** -- there is no translation at all, and `Untranslatable` says why.
* **diverged** -- there is one and the two engines disagree. The first position they
  disagree about is carried back in full, because a FEN and two move lists is the whole
  of what anybody needs to fix it.
* **verified** -- they offer the same legal moves over the tree and over the walk. It is
  not a proof, and it says nothing about results. The honest claim is the one the numbers
  support, so a `Verification` records how deep and how far it looked.

The engine side goes through a small protocol rather than a concrete class, because the
binary is not the only Fairy-Stockfish this can be checked against: `pyffish` on PyPI is
the same rules code as a library, it needs no binary at all, and it is what the committed
corpus test uses so that the test suite does not depend on somebody's ``PATH``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy import position as fairy_position
from chess_variant_sandbox.fairy.unread import reasons as unread_reasons
from chess_variant_sandbox.fairy.variant import Untranslatable, Variant, translate

# How deep the tree is compared, and how far the walk goes, when nobody says otherwise.
#
# Three plies is where the interesting disagreements live: one ply catches a piece that
# moves wrongly, two catches a move that should have been blocked, and three is the
# first depth at which a rule about what a move *leaves behind* -- an en passant window,
# a castling right, a piece in a hand -- can be counted. Deeper costs seconds and has
# never found anything the walk did not.
#
# Sixty plies is the opening and the early middle game, and some rules live later. Q438
# found shogi's first difference (a promotion on a move out of the zone) at a median of
# ply 108 of the walk, so a 60-ply walk passed it. A difference known to need that is
# refused in the translator rather than left to the walk to find.
DEFAULT_DEPTH = 3
DEFAULT_GAMES = 4
DEFAULT_PLIES = 60


class Backend(Protocol):
    """The four questions a verification asks the other engine."""

    def load(self, variant: Variant, config_path: str) -> None:
        ...

    def divide(self, fen: str, depth: int) -> dict[str, int]:
        ...

    def perft(self, fen: str, depth: int) -> int:
        ...

    def mated_after(self, fen: str, move: str) -> bool:
        """Whether ``move`` from ``fen`` leaves the side that replies in check with no move.

        Asked only about a move of theirs that ours lacks (`_tolerated`).
        """
        ...


@dataclass
class Verification:
    """What the two engines turned out to agree about."""

    ruleset_hash: str
    name: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    depth: int = 0
    positions: int = 0
    # The first position the two disagreed about, as a FEN, with both move lists.
    #
    # Kept rather than summarised: "the move counts differ at depth 3" is not something
    # anybody can act on, and the position that produced it is.
    fen: str = ""
    ours: list[str] = field(default_factory=list)
    theirs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Positions where theirs listed a mating drop of `Variant.drop_mate` that ours
    # forbids, and the rest of the two lists agreed (`_tolerated`, Q453).
    drop_mates: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.ruleset_hash,
            "name": self.name,
            "ok": self.ok,
            "reasons": self.reasons,
            "depth": self.depth,
            "positions": self.positions,
            "fen": self.fen,
            "ours": self.ours,
            "theirs": self.theirs,
            "notes": self.notes,
            "drop_mates": self.drop_mates,
        }

    def summary(self) -> str:
        if self.ok:
            return (
                f"verified to depth {self.depth} over {self.positions} positions"
                + (f"; {'; '.join(self.notes)}" if self.notes else "")
            )
        return "; ".join(self.reasons) or "diverged"


def verify(
    spec: dict[str, Any],
    backend: Backend,
    config_path: str,
    *,
    depth: int = DEFAULT_DEPTH,
    games: int = DEFAULT_GAMES,
    plies: int = DEFAULT_PLIES,
    seed: int = 0,
) -> Verification:
    """Play ``spec`` on both engines and report the first position whose legal moves differ.

    Moves only, never results: see the module docstring, and `results.check_results`.
    """
    try:
        variant = translate(spec)
    except Untranslatable as error:
        return Verification(
            ruleset_hash=_hash(spec),
            name=str(spec.get("name", "")),
            ok=False,
            reasons=list(error.reasons),
        )

    engine = Engine(spec)
    result = Verification(
        ruleset_hash=engine.spec.hash,
        name=engine.spec.name,
        ok=True,
        depth=depth,
        notes=list(variant.notes),
    )

    try:
        backend.load(variant, config_path)
    except Exception as error:  # noqa: BLE001 - any failure to load is a refusal
        result.ok = False
        result.reasons.append(f"the engine would not load the variant: {error}")
        return result

    # A key the build drops in silence changes the game it searches, often where no
    # walk can see it: a goal square decides the result, not the legal moves (Q915).
    # A backend that can say what it skipped (the binary, `UciEngine.unread_options`)
    # is asked first; `pyffish` reads every key and has no such question.
    asking = getattr(backend, "unread_options", None)
    found = asking(config_path) if callable(asking) else None
    if found:
        said = unread_reasons(variant.config, found)
        if said:
            result.ok = False
            result.reasons.extend(said)
            return result

    state = engine.initial_state()
    if not _agree_here(engine, variant, backend, state, result):
        return result

    # Generation 42 (N4c, `docs/archive/plan-npc.md` §4.6): a duck game's turn is two of our plies and
    # one move of theirs, so the ladder compares at turn ends only, our perft(2k) with their
    # perft(k), and a depth that ends mid-turn is rounded up to the turn's end.
    turns = engine.spec.duck is not None
    for level in range(1, (depth + 1) // 2 + 1 if turns else depth + 1):
        ours = engine.perft(state, 2 * level if turns else level)
        theirs = backend.perft(_fen(engine, variant, state), level)
        result.positions += ours
        if ours != theirs:
            result.ok = False
            result.fen = _fen(engine, variant, state)
            result.reasons.append(
                f"perft({level}) is {ours} here and {theirs} there"
            )
            return result

    _walk(engine, variant, backend, result, games=games, plies=plies, seed=seed)
    return result


def _walk(
    engine: Engine,
    variant: Variant,
    backend: Backend,
    result: Verification,
    *,
    games: int,
    plies: int,
    seed: int,
) -> None:
    """Play random games on our engine, asking theirs about every position.

    Random rather than searched, and that is the point: a search visits the positions
    it thinks are good, which are the positions the *evaluator* likes, and a rule it
    has never been shown would go unmet on every one of them. A random mover reaches
    promotion, drops and a bare board, which is where the move rules that are not about
    geometry live.

    It stops at the first position our engine calls over, before asking theirs. So a
    mate, a stalemate, a draw or a flag is reached but never compared: who won is not
    this function's question.
    """
    for game in range(games):
        rng = random.Random((seed << 8) ^ game)
        state = engine.initial_state()
        for _ in range(plies):
            legal = engine.legal_moves(state)
            if not legal or engine.terminal(state, legal) is not None:
                break
            # A duck game is compared at turn starts only: a position owing the duck's half
            # has no FEN there, where a move is both halves.
            if not engine.duck_owed(state) and not _agree_here(
                engine, variant, backend, state, result
            ):
                return
            state = engine.apply(state, rng.choice(legal))


def _agree_here(
    engine: Engine,
    variant: Variant,
    backend: Backend,
    state: GameState,
    result: Verification,
) -> bool:
    """Whether the two engines offer the same moves in this one position."""
    text = _fen(engine, variant, state)
    enders: set[str] = set()
    if engine.spec.duck is not None:
        ours, enders = _turn_words(engine, variant, state)
    else:
        ours = sorted(
            fairy_position.uci_of(
                engine.spec, state, move, variant.chars, variant.promotion_suffix
            )
            for move in engine.legal_moves(state)
        )
    if len(set(ours)) != len(ours):
        result.ok = False
        result.fen = text
        result.ours = ours
        result.reasons.append(
            "two of our moves have the same spelling there, so the opponent could not "
            "name one of them"
        )
        return False
    try:
        theirs = sorted(backend.divide(text, 1))
    except Exception as error:  # noqa: BLE001
        result.ok = False
        result.fen = text
        result.reasons.append(f"the engine failed on this position: {error}")
        return False
    result.positions += 1
    if enders:
        # A piece move that ends our game has no duck half: it stands for every word of
        # theirs that starts with it, and there must be at least one.
        ended = [word for word in theirs if word.split(",")[0] in enders]
        missing = enders - {word.split(",")[0] for word in ended}
        theirs = sorted(set(theirs) - set(ended))
        ours = sorted([*ours, *missing])
    extra = _tolerated(variant, backend, text, ours, theirs)
    if extra:
        result.drop_mates += 1
        theirs = sorted(set(theirs) - extra)
    if ours != theirs:
        result.ok = False
        result.fen = text
        result.ours = ours
        result.theirs = theirs
        only_ours = sorted(set(ours) - set(theirs))
        only_theirs = sorted(set(theirs) - set(ours))
        result.reasons.append(
            "the move lists differ: "
            + (f"we have {only_ours} " if only_ours else "")
            + (f"they have {only_theirs}" if only_theirs else "")
        )
        return False
    return True


def _turn_words(
    engine: Engine, variant: Variant, state: GameState
) -> tuple[list[str], set[str]]:
    """Our turns from a duck game's turn start, as their words (generation 42, N4c, §4.6).

    Each piece move with each duck move our engine then offers (`position.turn_word`),
    read ply by ply off our engine; and apart, the piece moves that end our game, which
    have no duck half here and are matched by prefix (`_agree_here`).
    """
    words: list[str] = []
    enders: set[str] = set()
    spec, chars, suffix = engine.spec, variant.chars, variant.promotion_suffix
    for move in engine.legal_moves(state):
        half = engine.apply(state, move)
        if engine.terminal(half) is not None:
            enders.add(fairy_position.turn_word(spec, state, move, None, chars, suffix))
            continue
        for duck in engine.legal_moves(half):
            words.append(fairy_position.turn_word(spec, state, move, duck, chars, suffix))
    return sorted(words), enders


def _tolerated(
    variant: Variant, backend: Backend, fen: str, ours: list[str], theirs: list[str]
) -> set[str]:
    """Their extra moves that are the one difference the translation knows of, or none.

    ``drops.no_mate`` forbids a drop that mates, and ours leaves it out of the legal
    moves. Theirs (``shogiPawnDropMateIllegal``) lists it and scores it as a loss for the
    side that plays it (`variant._check_drop_mate`). So theirs may have extra moves here,
    and they are accepted only when every one of them is a drop of `Variant.drop_mate`
    after which *their* engine says the side to reply is in check with no move, and
    ours has nothing theirs lacks. Anything else returns nothing and the lists are
    compared as they stand. Their engine answers whether the drop mates, not Python
    (architecture rule 3). The perft ladder from the opening gets no such allowance: a
    hand is empty there for at least two plies, and a count cannot say which move it is.
    """
    if not variant.drop_mate:
        return set()
    extra = set(theirs) - set(ours)
    if not extra or not set(ours) <= set(theirs):
        return set()
    drop = f"{variant.drop_mate}@"
    if not all(move.startswith(drop) for move in extra):
        return set()
    if not all(backend.mated_after(fen, move) for move in sorted(extra)):
        return set()
    return extra


def _fen(engine: Engine, variant: Variant, state: GameState) -> str:
    return fairy_position.variant_fen(variant, engine.spec, state)


def _hash(spec: dict[str, Any]) -> str:
    from chess_variant_sandbox.spec.canonical import spec_hash

    try:
        return spec_hash(spec)
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- backends


class PyffishBackend:
    """Fairy-Stockfish as a library, for checking without a binary.

    ``pyffish`` is the same rules code the engine is built from, published by the same
    author, and it answers every question a verification asks. What it cannot do is
    *search*, which is why it is not also the opponent.

    It is a C extension with a C extension's manners: a malformed variant or a position
    that does not match it can end the process rather than raise. Everything it is
    given here has already been through `translate`, and the corpus test that drives it
    is the one place a bad config could arrive -- which is a reason to read a failure
    here as a translation bug rather than as a flaky test.
    """

    def __init__(self) -> None:
        import pyffish  # noqa: PLC0415 - optional, and only where it is used

        self._sf = pyffish
        self._name = ""
        self._insufficient = False

    def load(self, variant: Variant, config_path: str) -> None:
        self._sf.load_variant_config(variant.config)
        if variant.name not in self._sf.variants():
            raise RuntimeError(f"pyffish did not register {variant.name!r}")
        self._name = variant.name
        self._insufficient = variant.insufficient_material

    def divide(self, fen: str, depth: int) -> dict[str, int]:
        moves = self._sf.legal_moves(self._name, fen, [])
        if depth <= 1:
            return dict.fromkeys(moves, 1)
        return {
            move: self.perft(self._sf.get_fen(self._name, fen, [move]), depth - 1)
            for move in moves
        }

    def perft(self, fen: str, depth: int) -> int:
        if depth <= 0:
            return 1
        moves = self._sf.legal_moves(self._name, fen, [])
        if depth == 1:
            return len(moves)
        return sum(
            self.perft(self._sf.get_fen(self._name, fen, [move]), depth - 1)
            for move in moves
        )

    # The result check's two questions (`results.Referee`, Q404). The UCI binary has no
    # way to ask either, which is why the result check uses the library only.

    def ending(self, fen: str, moves: Sequence[str]) -> tuple[str, int] | None:
        """Whether the game is over after ``moves`` from ``fen``, and how it stands.

        ``None`` while it goes on. Otherwise how it ended and what it is worth to the
        side to move then, in the engine's units: positive wins, negative loses, zero
        draws. Asked in the order their search asks: an ending the variant declares, a
        draw a player may claim, then no legal move. Each call builds the position again
        and costs about 6 ms here, so no question is asked that the answer does not
        need.
        """
        played = list(moves)
        ended, value = self._sf.is_immediate_game_end(self._name, fen, played)
        if ended:
            return "ended", int(value)
        # Q1069a: a dead position, where the ruleset draws by insufficient material: both
        # sides short of mating material, as their `has_insufficient_material` reads it.
        if self._insufficient and all(
            self._sf.has_insufficient_material(self._name, fen, played)
        ):
            return "insufficient material", 0
        claimable, value = self._sf.is_optional_game_end(self._name, fen, played)
        if claimable:
            return "a draw may be claimed", int(value)
        if not self._sf.legal_moves(self._name, fen, played):
            return "no legal move", int(self._sf.game_result(self._name, fen, played))
        return None

    def fen_after(self, fen: str, moves: Sequence[str]) -> str:
        """The FEN their engine reaches by playing ``moves`` from ``fen`` itself."""
        return str(self._sf.get_fen(self._name, fen, list(moves)))

    def mated_after(self, fen: str, move: str) -> bool:
        """`Backend.mated_after`: in check after ``move``, and no legal reply."""
        played = [move]
        if self._sf.legal_moves(self._name, fen, played):
            return False
        return bool(self._sf.gives_check(self._name, fen, played))
