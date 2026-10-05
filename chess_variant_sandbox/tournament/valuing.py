"""Running the search: hill-climb a piece-value table against playing strength itself.

``agents/valuations.py`` holds the procedure's constants and the store; this is the pass
that executes it, and it lives under ``tournament/`` for the reason the fit does -- it
plays games, and ``agents/`` must not.

**Coordinate ascent, one piece at a time, on a log scale.** Multiply one piece's value by
a step, ask ``duel`` whether the resulting player beats the one it came from, keep it if
so, and move on. Coarse steps first, then fine. Cheap, interpretable -- every accepted
step is a sentence about one piece -- and it cannot produce a negative value unless a
negative value genuinely wins, which the deleted regression could and did.

**Nothing is seeded with human values.** Classical chess values lose at this budget,
repeatably and at three seeds, so they are not ground truth; they are another hypothesis,
and one this project has already measured. The search starts from the derived table,
which is the thing it has to beat.

**Two seeds, and which is which is the whole of the method.** The search holds one seed
set fixed across every comparison -- that is what common random numbers means, and it is
what makes two tables differing in one piece comparable at all. Holding it fixed also
means the search can *fit* it. So the verdict is taken on a different seed set, and
``Valuation.accept_score`` is that held-out number rather than the best thing the search
saw.

**What a candidate is spelled as matters more than it looks.** Every candidate plays
under ``values=<digest>``, resolved through the real store, so the search optimises
exactly the player the store will ship -- including the fact that a player with an
opinion about the pieces is a ``FeatureEvaluator`` and not ``eval::Evaluator``, which is
a decision the core makes deliberately and this stage inherits rather than works around.

**And so does what the verdict compares it against.** The reference is the *derived*
table played by that same kind of player, so the two arms differ in the valuation and in
nothing else, and a win is attributable to it. The searched table is played against
``minimax:nodes=N`` as well -- the rung everything else in the project is about -- and
that number is reported beside the verdict rather than gated on, because it crosses two
differences at once: the values, and the built-in evaluator's own arithmetic, which
``builtin_vector`` reproduces in its decisions but not in its bits.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.features import builtin_vector, feature_context
from chess_variant_sandbox.agents.valuations import (
    SEARCH,
    SearchProcedure,
    Valuation,
    register_valuation,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import CompiledSpec
from chess_variant_sandbox.spec.canonical import HASH_LENGTH
from chess_variant_sandbox.tournament.duel import Duel, duel

# Budgets the plies probe reports at. Nothing in this stage reads them; they are what a
# depth-aware valuation needs, and measuring them costs a fraction of one game. Spanning
# the rated ladder and one rung past it, because the pilots found the effect growing with
# the budget past the top of that ladder.
PROBE_BUDGETS = (400, 1_600, 6_400, 25_600)


@dataclass(frozen=True)
class _Arms:
    """The three things every candidate on one ruleset shares.

    Bundled because they travel together to every helper here and a helper taking six
    positional arguments is a helper whose call sites can transpose two of them -- which
    with two same-shaped value tables in the list is precisely the failure ``PieceValues``
    exists to prevent one level down.
    """

    compiled: CompiledSpec
    derived: tuple[float, ...]
    weights: tuple[float, ...]


def search_valuation(
    spec: dict[str, Any],
    procedure: SearchProcedure = SEARCH,
    workers: int = 0,
    progress: bool = False,
) -> Valuation:
    """Search this ruleset's piece values, check the result, and return the verdict.

    Deterministic in ``(spec, procedure)`` alone: the budget, the seeds, the steps, the
    order the pieces are visited in and the acceptance bar are all constants of the
    procedure. Two machines running this produce the same table, which is what makes a
    searched player reproducible from the repository rather than from somebody's cache.
    """
    engine = Engine(spec)
    compiled = engine.spec
    arms = _Arms(
        compiled=compiled,
        derived=tuple(Evaluator.for_spec(compiled).values),
        weights=builtin_vector(feature_context(compiled)),
    )

    # Royal slots are pinned at zero by `ROYAL_VALUE` and refused if set, and a
    # multiplicative step cannot move a zero anywhere. Both are skipped for one reason
    # rather than two: a piece whose derived value is zero is one this parameterisation
    # has nothing to say about.
    movable = tuple(
        piece_type
        for piece_type in range(compiled.n_types)
        if piece_type not in compiled.royal_types and arms.derived[piece_type] > 0.0
    )

    incumbent = list(arms.derived)
    comparisons = 0
    accepted_steps = 0
    exhausted = False

    for step in procedure.steps:
        if exhausted:
            break
        for _sweep in range(procedure.sweeps):
            improved = False
            for piece_type in movable:
                if comparisons >= procedure.max_comparisons:
                    exhausted = True
                    break
                spent, taken = _try_piece(
                    spec, arms, incumbent, piece_type, step, procedure, workers, progress
                )
                comparisons += spent
                if taken is not None:
                    incumbent = taken
                    accepted_steps += 1
                    improved = True
            if exhausted or not improved:
                break

    return _verdict(
        spec,
        engine,
        arms,
        tuple(incumbent),
        procedure,
        comparisons,
        accepted_steps,
        exhausted,
        workers,
        progress,
    )


def _try_piece(
    spec: dict[str, Any],
    arms: _Arms,
    incumbent: list[float],
    piece_type: int,
    step: float,
    procedure: SearchProcedure,
    workers: int,
    progress: bool,
) -> tuple[int, list[float] | None]:
    """Try one piece up and down. Returns how many comparisons it cost and the winner.

    Up first, and it stops at the first direction that clears the bar rather than playing
    both and taking the better. A step is a hypothesis about one piece and the other
    direction is a different hypothesis, so choosing between them would need a comparison
    between them -- and taking the better of two noisy numbers is exactly the selection
    bias a paired estimator exists to avoid re-introducing.
    """
    reference = _table_agent(arms, tuple(incumbent), procedure.nodes)
    spent = 0
    for factor in (step, 1.0 / step):
        candidate = list(incumbent)
        candidate[piece_type] = incumbent[piece_type] * factor
        result = duel(
            spec,
            _table_agent(arms, tuple(candidate), procedure.nodes),
            reference,
            procedure.pairs,
            procedure.seed,
            workers,
        )
        spent += 1
        taken = _improved(result, procedure)
        if progress:
            print(
                f"    {arms.compiled.piece_ids[piece_type]:<3} x{factor:<5.3f}  "
                f"{result.score:.4f} +/-{result.error:.4f}  "
                f"identical {result.identical}/{len(result.pairs)}  "
                f"{'kept' if taken else ''}",
                flush=True,
            )
        if taken:
            return spent, candidate
    return spent, None


def _improved(result: Duel, procedure: SearchProcedure) -> bool:
    """Whether a comparison says the candidate is better, at the sample's own error.

    ``accept_sigma`` standard errors above 0.5, and the standard error is the one this
    sample measured rather than a binomial constant -- so a step where the two players
    mostly agreed, and the pairing cancelled most of the noise, is held to a
    correspondingly tighter bar than one where they disagreed everywhere.
    """
    return result.score - 0.5 > procedure.accept_sigma * result.error


def _verdict(
    spec: dict[str, Any],
    engine: Engine,
    arms: _Arms,
    vector: tuple[float, ...],
    procedure: SearchProcedure,
    comparisons: int,
    accepted_steps: int,
    exhausted: bool,
    workers: int,
    progress: bool,
) -> Valuation:
    """Play the held-out acceptance matches and assemble the store entry.

    A search that moved nothing is refused without a match. It *is* the derived table, so
    the match would spend a hundred games to report exactly 0.5 -- and a 0.5 reported as
    a refusal would be indistinguishable from a search that tried something and lost.
    """
    plies = _plies(engine, procedure)
    if vector == arms.derived:
        return _entry(
            arms, vector, procedure, comparisons, accepted_steps, exhausted, plies,
            accepted=False, accept=None, ladder=None,
        )

    provisional = _entry(
        arms, vector, procedure, comparisons, accepted_steps, exhausted, plies,
        accepted=True, accept=None, ladder=None,
    )
    # Registered before the matches because they play it *by its real name*, which is the
    # discipline `register_fit(..., replace=True)` keeps and for the same reason: testing
    # a different spelling would leave the resolution path itself untested, and that is
    # where a searched player is most likely to be quietly wrong.
    register_valuation(provisional, replace=True)
    searched = f"minimax:values={procedure.digest},nodes={procedure.accept_nodes}"

    if progress:
        print(f"  acceptance match at {procedure.accept_nodes} nodes", flush=True)
    accept = duel(
        spec,
        searched,
        # The derived table played by the same kind of player, so the two arms differ in
        # the valuation and in nothing else.
        _table_agent(arms, arms.derived, procedure.accept_nodes),
        procedure.accept_pairs,
        procedure.accept_seed,
        workers,
    )
    ladder = duel(
        spec,
        searched,
        f"minimax:nodes={procedure.accept_nodes}",
        procedure.accept_pairs,
        # A different seed again, so the two reported numbers are not two readings of one
        # sample dressed as two measurements.
        procedure.accept_seed + 1,
        workers,
    )

    final = _entry(
        arms, vector, procedure, comparisons, accepted_steps, exhausted, plies,
        # Strictly better, which is the word the fit's bar was written in and the same
        # bar. A margin here would be one chosen after seeing which rulesets it excludes;
        # the measured error travels beside the verdict instead, so a reader can see
        # which verdicts the sample could resolve.
        accepted=accept.score > 0.5,
        accept=accept,
        ladder=ladder,
    )
    register_valuation(final, replace=True)
    return final


def _entry(
    arms: _Arms,
    vector: tuple[float, ...],
    procedure: SearchProcedure,
    comparisons: int,
    accepted_steps: int,
    exhausted: bool,
    plies: dict[str, float],
    accepted: bool,
    accept: Duel | None,
    ladder: Duel | None,
) -> Valuation:
    """One store entry, so the provisional and the final one cannot drift apart."""
    from chess_variant_sandbox.measurement import built_fingerprint

    return Valuation(
        ruleset_hash=arms.compiled.hash,
        procedure=procedure.digest,
        piece_ids=tuple(arms.compiled.piece_ids),
        vector=vector,
        derived=arms.derived,
        weights=arms.weights,
        accepted=accepted,
        accept_score=round(accept.score, 4) if accept else 0.5,
        accept_error=round(accept.error, 4) if accept else 0.0,
        accept_games=accept.games if accept else 0,
        ladder_score=round(ladder.score, 4) if ladder else 0.5,
        ladder_error=round(ladder.error, 4) if ladder else 0.0,
        comparisons=comparisons,
        accepted_steps=accepted_steps,
        exhausted=exhausted,
        plies=plies,
        # Stamped here rather than at the top of the search, so it names the core that
        # played the acceptance match -- which is the number a reader is trusting.
        build=built_fingerprint(),
    )


def _plies(engine: Engine, procedure: SearchProcedure) -> dict[str, float]:
    """Plies completed per search at each probe budget, by the heuristic pass's own probe.

    The same measurement ``heuristics._depth_reached`` takes, on the plain ladder agent
    rather than on a heuristic one, so the numbers are comparable with
    ``docs/reference/search-depth.md``. Nothing in Stage B reads them; a depth-aware
    valuation needs them, and taking them here is what makes that stage cost no games.
    """
    from chess_variant_sandbox.tournament.heuristics import plies_completed

    return {
        str(budget): plies_completed(engine, f"minimax:nodes={budget}", procedure.seed)
        for budget in PROBE_BUDGETS
    }


def _table_agent(arms: _Arms, table: tuple[float, ...], nodes: int) -> str:
    """A player on this exact table, addressable by name and resolved through the store.

    The name is the table's own content hash standing in a procedure digest's place,
    which is a *transient* registration and never written to disk: during a search there
    are hundreds of tables and only one procedure, so the store's key cannot address them
    and the agent string has to. What it buys is that a candidate is resolved, and
    therefore played, by exactly the code the shipped player uses.
    """
    digest = _table_digest(arms.compiled.hash, table)
    register_valuation(
        Valuation(
            ruleset_hash=arms.compiled.hash,
            procedure=digest,
            piece_ids=tuple(arms.compiled.piece_ids),
            vector=table,
            derived=arms.derived,
            weights=arms.weights,
            accepted=True,
            accept_score=0.0,
            accept_error=0.0,
            accept_games=0,
        ),
        replace=True,
    )
    return f"minimax:values={digest},nodes={nodes}"


def _table_digest(ruleset_hash: str, table: tuple[float, ...]) -> str:
    """A content hash of one table, at the precision two tables are the same player at.

    The ruleset hash is mixed in so a transient name from one search cannot collide with
    one from another inside a process that ran both.
    """
    blob = json.dumps(
        {"ruleset": ruleset_hash, "values": [round(v, 9) + 0.0 for v in table]},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:HASH_LENGTH]


def timed_search(
    spec: dict[str, Any],
    procedure: SearchProcedure = SEARCH,
    workers: int = 0,
    progress: bool = False,
) -> tuple[Valuation, float]:
    """``search_valuation`` with the wall clock, which is what a campaign log wants."""
    started = time.perf_counter()
    valuation = search_valuation(spec, procedure, workers, progress)
    return valuation, time.perf_counter() - started
