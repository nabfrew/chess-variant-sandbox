"""The agent protocol and the registry that turns a config string into a player.

Two conventions here are load-bearing for the rest of the project.

**Budget is nodes, never depth.** Depth is not comparable across rulesets: a game with
a branching factor of 8 and one with a branching factor of 60 do very different amounts
of work at "depth 4". Since the index measures how much stronger an agent gets as it is
allowed to think more, the thinking allowance has to mean the same thing everywhere.

**Budget is part of agent identity.** ``minimax:nodes=20000`` and ``minimax:nodes=2000``
are two different players with two different ratings. That is exactly the comparison
the strategic-depth metric reads, so it must survive into the tournament record rather
than living in a runner's local variable.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from chess_variant_sandbox.engine import Engine, GameState


@dataclass
class SearchStats:
    """Per-game counters, recorded alongside results so cost is visible next to strength.

    ``depth_sum`` and ``searches`` are the pair that says how deep a rung actually saw.
    A node budget is not a fixed amount of search -- at 6,400 nodes the ``dull`` control
    reaches 8.5 plies and xiangqi reaches 2.0 -- so a ladder read as though its rungs
    were the same players is reading the branching factor. Two numbers rather than a
    mean, because a mean over one game cannot be pooled over a corpus.

    Both stay at zero for an agent that does not deepen, and a position with a single
    legal move contributes to neither: no search happened there.
    """

    nodes: int = 0
    moves: int = 0
    depth_sum: int = 0
    searches: int = 0

    def merge(self, other: SearchStats) -> None:
        self.nodes += other.nodes
        self.moves += other.moves
        self.depth_sum += other.depth_sum
        self.searches += other.searches


class Agent(Protocol):
    """A player. One method, and it must not touch global state.

    Every source of randomness arrives through ``rng``. That is what makes a game
    reproducible from ``(ruleset_hash, seed, agent_configs)``, and reproducibility is
    what lets a surprising tournament result be investigated rather than re-rolled.
    """

    name: str

    # Declared on the protocol rather than left to each implementation: the game runner
    # reads it to record search cost alongside the result, so an agent without it would
    # play fine and then break the record writer.
    stats: SearchStats

    def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
        ...


@dataclass(frozen=True)
class AgentConfig:
    """A parsed ``kind:key=value,...`` specification."""

    kind: str
    options: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.options:
            return self.kind
        rendered = ",".join(f"{k}={v}" for k, v in sorted(self.options.items()))
        return f"{self.kind}:{rendered}"


AgentBuilder = Callable[[Engine, dict[str, Any]], Agent]
_REGISTRY: dict[str, AgentBuilder] = {}

# The options the Rust core understands, in the fixed order it reads them. Every native
# agent fills the whole tuple and ignores the slots that do not apply to it: a fixed
# shape means adding an option is a change in two files rather than a new FFI signature.
#
# ``(kind, nodes, mobility, max_depth, rollout_depth, exploration, goal, weights,
#    values)``.
#
# The seventh is stage 17A's: what a whole win by a non-capture ending is worth, in mean
# non-royal pieces. It is an agent option rather than a constant in the core because the
# sweep that fixed it had to be able to turn it off, and because ``goal=0`` is the only
# honest control for "did the term do anything on this ruleset".
#
# The eighth is stage 19's: a heuristic's weight vector over ``features.VOCABULARY``, and
# an **empty tuple means the built-in objective** -- which is what keeps every committed
# number where it is, because every built-in agent sends one.
#
# The ninth is stage 30's: what this player thinks the pieces are worth, ruleset-shaped
# rather than vocabulary-shaped, empty for the table the rules imply. It rides here
# *beside* the weights and does not replace the ``values`` argument every core call
# already takes -- that one is the derived table, which normalises material and orders
# captures for every player whatever its opinion. Nine is comfortably inside PyO3's
# twelve-element tuple conversion; the ceiling CLAUDE.md warns about is the *state*
# tuple's, not this one's.
#
# The tenth is Q466's: the immunity term's weight, in mean non-royal pieces an immune
# piece (``evaluation.Evaluator.immunity``). ``IMMUNE_PIECES`` for a greedy or minimax
# player that does not spell ``immune=`` (Q471), always sent explicitly so the core holds
# no default of its own; ``0.0`` for the players with no evaluator.
#
# The eleventh is Q502e's: a learned net's store bytes (``agents/nets.py``), ``b""`` for
# every player without one. A ``net=`` player whose store is absent, refused or unreadable
# never gets here: ``nets.require_net`` refuses it (Q709), because ``b""`` is the built-in
# objective and would play it under the learned player's name.
#
# The twelfth is Q502i's: the search repairs as bits (``minimax.KEEP_TABLE``,
# ``THREATS``, ``LOUD_DROPS``), ``0`` for today's search and for every player that is not
# minimax. Twelve is PyO3's ceiling, which is why three flags share one slot.
NativeSpec = tuple[
    str, int, float, int, int, float, float, tuple[float, ...], tuple[float, ...], float, bytes,
    int,
]

# Where the two objective halves sit, named so a reader is not counting commas and a
# test is not writing `[-1]` -- which meant the weights until the values were added
# after them, and then silently meant something else.
NATIVE_WEIGHTS = 7
NATIVE_VALUES = 8

# A resolver sees the options and the hash of the ruleset the agent is about to play.
#
# The second argument arrived with the fitted objective and is not decoration. A stated
# heuristic resolves through a content-addressed digest and needs no ruleset -- the
# digest *is* the vector. A *fitted* objective is the opposite by design: the name
# carries the procedure, which is the same string on every board, and the vector it
# produces is per-ruleset, so resolution cannot happen without knowing which board.
# Every caller already holds an engine; the hash travels rather than a global binding,
# because a module-level "current ruleset" would be wrong the moment two rulesets are
# open at once, and it would be wrong silently.
NativeResolver = Callable[[dict[str, Any], str], NativeSpec]
_NATIVE: dict[str, NativeResolver] = {}


def register(kind: str) -> Callable[[AgentBuilder], AgentBuilder]:
    def decorator(builder: AgentBuilder) -> AgentBuilder:
        _REGISTRY[kind] = builder
        return builder

    return decorator


def register_native(kind: str) -> Callable[[NativeResolver], NativeResolver]:
    """Declare that the Rust core can run this agent kind.

    The resolver lives next to the Python builder it mirrors, so the two cannot drift
    on a default. An agent registered with ``register`` but not here still plays
    perfectly well -- it just keeps the whole game in Python, which is the escape hatch
    that makes ``register`` a real extension point rather than a list of four names.
    """

    def decorator(resolver: NativeResolver) -> NativeResolver:
        _NATIVE[kind] = resolver
        return resolver

    return decorator


def native_spec(config: AgentConfig | str, ruleset_hash: str = "") -> NativeSpec | None:
    """The core's parameters for ``config``, or ``None`` if it has no native form.

    ``ruleset_hash`` defaults to empty because most agents do not need it and every
    caller that does not pass one is asking about an agent that cannot depend on the
    board. An agent that *does* -- a fitted objective -- raises rather than resolving to
    something, so the default cannot quietly produce the wrong player.
    """
    if isinstance(config, str):
        config = parse_agent(config)
    resolver = _NATIVE.get(config.kind)
    return None if resolver is None else resolver(dict(config.options), ruleset_hash)


def available_agents() -> list[str]:
    return sorted(_REGISTRY)


def parse_agent(text: str) -> AgentConfig:
    """Parse ``"minimax:nodes=5000,mobility=0.1"`` into a config.

    Values are coerced to int, then float, then left as strings, so a config round-trips
    through ``str()`` unchanged and two spellings of the same agent produce the same
    identity in a tournament record.
    """
    kind, _, remainder = text.partition(":")
    kind = kind.strip()
    if kind not in _REGISTRY:
        raise ValueError(f"unknown agent {kind!r}; available: {', '.join(available_agents())}")

    options: dict[str, Any] = {}
    for chunk in filter(None, (c.strip() for c in remainder.split(","))):
        key, _, value = chunk.partition("=")
        if not value:
            raise ValueError(f"agent option {chunk!r} must be written key=value")
        options[key.strip()] = _coerce(value.strip())
    return AgentConfig(kind=kind, options=options)


def build_agent(config: AgentConfig | str, engine: Engine) -> Agent:
    """Instantiate an agent for one engine. Agents are per-ruleset, never shared."""
    if isinstance(config, str):
        config = parse_agent(config)
    return _REGISTRY[config.kind](engine, dict(config.options))


def _coerce(value: str) -> Any:
    """``value`` as the number it spells -- but only when that spelling survives.

    The coercion exists for the two reasons the docstring above gives, and both of them
    are the *same* reason: a config has to round-trip through ``str()`` unchanged. A
    coercion that does not round-trip fails at the only job it has, and two of them
    reach here through ``h=<digest>``, where the value is an identifier that merely
    looks like a number:

    * ``096717939346`` -- all digits, so ``int`` takes it and the leading zero is gone.
      One heuristic digest in sixteen starts with a zero.
    * ``37625e727351`` -- valid float syntax, so ``float`` takes it, the exponent
      overflows, and a twelve-character digest becomes ``inf``. Roughly one digest in
      two hundred is shaped like that, which over a corpus this size is better than
      even odds of meeting one -- and the first model-proposed set in the project
      contained one.

    ``normalise_digest`` repairs the first after the fact and cannot repair the second,
    because ``float`` has already thrown the digits away. Asking whether the value
    survives the trip is the general form of both, and it costs one comparison.
    """
    for caster in (int, float):
        try:
            number = caster(value)
        except ValueError:
            continue
        # A coercion that changes the text is a coercion that loses the identity.
        return number if str(number) == value else value
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value
