"""Which objective an agent's options name, when there are now three ways to name one.

There are exactly four answers and no two of them may be reachable at once:

* ``h=<digest>`` -- a *stated* vector, content-addressed, optionally carrying its own
  piece values. Per-ruleset by construction, never in the rated roster, and resolved
  without knowing the board because the digest is the vector.
* ``fit=<procedure>`` -- a *fitted* vector, procedure-addressed. The name is the same
  string on every board and the vector is derived per ruleset, which is what lets it
  into the roster at all; resolving it therefore needs the ruleset.
* ``values=<procedure>`` -- a *searched* piece-value table, procedure-addressed the same
  way and resolved the same way. It states no opinion about the weights, so it carries
  the vector that reproduces the built-in objective and differs from it in the price list
  alone.
* none of them -- the built-in spec-derived evaluator on the derived table, spelled as
  two empty tuples, which is what keeps every committed number where it is.

**Any two at once is refused rather than ordered.** A precedence rule here would be a
silent answer to a question the caller got wrong, and the failure it hides is the worst
kind this project has: an agent playing a different objective under an unchanged name.

``values=`` is refused alongside the other two for a reason worth stating, because the
combination with ``fit=`` is the one that looks harmless -- a fitted weight vector prices
*how much material matters* and a searched table prices *which pieces*, so they are not
obviously the same claim. They are refused together anyway, because both were searched or
fitted against the built-in evaluator standing in the other's place, and a player made of
two halves neither of which was measured beside the other is a player nobody has
measured. If a stage ever wants that combination, the way in is a procedure that fits
both at once and carries one digest, exactly as ``fit=`` is one digest and not fifteen.
"""

from __future__ import annotations

from typing import Any

from chess_variant_sandbox.agents.fitted import fitted_weights
from chess_variant_sandbox.agents.heuristics import values_for, weights_for
from chess_variant_sandbox.agents.valuations import searched_values, searched_weights

# The option names that each claim to say where an objective's numbers came from.
# Q502e's `net=` is a fourth: a learned net replaces the whole objective.
_OBJECTIVE_OPTIONS = ("h", "fit", "values", "net")


def _refuse_two(options: dict[str, Any]) -> None:
    named = [key for key in _OBJECTIVE_OPTIONS if options.get(key) is not None]
    if len(named) > 1:
        raise ValueError(
            f"an agent names {' and '.join(named)} at once; each of "
            f"{', '.join(_OBJECTIVE_OPTIONS)} is a different claim about where this "
            "player's numbers came from, and holding two of them means one is wrong"
        )


def objective_weights(options: dict[str, Any], ruleset_hash: str = "") -> tuple[float, ...]:
    """The weight vector these options name, or ``()`` for the built-in objective.

    ``values=`` returns ``features.builtin_vector`` for the ruleset, which is not a
    choice this option makes. The core sends an empty weight vector to
    ``eval::Evaluator``, which reads the derived table and only the derived table, so a
    player with an opinion about the pieces has to be a feature evaluator and needs the
    vector that makes one behave like the built-in objective. That vector is
    ruleset-dependent and resolution holds only a hash, so the search writes it into the
    store; see ``valuations.Valuation.weights``.
    """
    _refuse_two(options)
    if options.get("fit") is not None:
        return fitted_weights(options, ruleset_hash)
    if options.get("values") is not None:
        return searched_weights(options, ruleset_hash)
    return weights_for(options)


def objective_values(options: dict[str, Any], ruleset_hash: str = "") -> tuple[float, ...]:
    """The piece-value override these options name, or ``()`` for the derived table.

    The sibling of ``objective_weights``, refusing the same combinations. Two functions
    rather than one returning a pair because every native resolver builds a fixed-shape
    tuple slot by slot, and a resolver that unpacked a pair into two slots is a resolver
    that can put them in the wrong order.

    **A fitted objective states nothing about the pieces.** Stage 30B's procedure that did
    fit them lost its measurement and was deleted, so ``fit=`` falls through to the
    derived table here. ``values=`` is the replacement and it optimises playing strength
    directly rather than the outcome of a game.
    """
    _refuse_two(options)
    if options.get("values") is not None:
        return searched_values(options, ruleset_hash)
    return values_for(options)
