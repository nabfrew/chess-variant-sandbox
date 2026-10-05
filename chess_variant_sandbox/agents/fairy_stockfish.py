"""Fairy-Stockfish as an opponent: the one player in the roster we did not write.

Every strength number this project reports is the roster grading itself. Elo is a
relative scale, `skill_gradient` is one rung against another, and `ladder_monotonicity`
asks whether our own search gets better when given more of our own nodes. None of that
can tell a ladder that is uniformly strong from one that is uniformly weak, and the
index's whole claim rests on the difference. An engine somebody else wrote, offering the
same legal moves, is the only instrument available for that -- which is why this agent
exists and why it is worth the translation layer under `chess_variant_sandbox/fairy/` that makes it
possible. The moves are what `fairy/verify.py` checks. Who has won is only as right as
`fairy/variant.py` reads the ending rules (`docs/reference/fairy-stockfish.md`).

Three things about it are unlike every other agent here, and each is deliberate.

**It is not a native agent.** There is no `register_native`, so the core knows nothing
about it and `play_game` cannot hand it the whole game. That is exactly the extension
point `CLAUDE.md` describes -- an agent the core knows nothing about still plays, over
Rust movegen -- and this is the first thing in the project to use it. It is also what
forced `play_game_mixed`: falling all the way back to `play_game_python` would have run
the *opponent* from `agents/` too, and those are declared oracles a hundred times slower
than the core.

**It is not deterministic in the way the rest of the corpus is.** A game here is a pure
function of `(ruleset, seed, agents)` because every agent's randomness comes through
`rng`; Fairy-Stockfish has its own search and its own tie-breaks, and a node-limited
search is reproducible on one build and not across two. So a result against it is not a
corpus measurement, and nothing under `metrics/` may read one. It belongs in the
picker's own on-demand table, which is already the table that rates players the rated
ladder must not contain -- see `web/opponents.py`.

**It can refuse to play.** No binary, or a ruleset with no counterpart, and there is no
agent to build. `available()` and `supports()` are asked before it is offered, and
`fairy.MoveMismatch` is what happens if the translation was wrong anyway: the game
stops rather than continuing under rules nobody wrote.
"""

from __future__ import annotations

import contextlib
import random
from typing import Any

from chess_variant_sandbox.agents.base import SearchStats, register
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.fairy import (
    FairyEngineError,
    MoveMismatch,
    UciEngine,
    Untranslatable,
    config_path,
    translate,
)
from chess_variant_sandbox.fairy import available as _binary_available
from chess_variant_sandbox.fairy import position as fairy_position
from chess_variant_sandbox.fairy import supports as _translates

# What the opponent searches when nobody says otherwise.
#
# Matched to `web/opponents.LADDER`'s top rung rather than chosen for strength: the
# question the picker's rating table answers is "how do our players compare with an
# outside one", and the only honest way to ask it is at a budget one of ours also gets.
# It is not a fair fight even so -- their node is a much better node than ours -- and
# that gap is the measurement rather than a flaw in it.
DEFAULT_NODES = 200_000

__all__ = ["DEFAULT_NODES", "FairyStockfishAgent", "available", "supports"]


def available() -> bool:
    """Whether a Fairy-Stockfish this project can drive is installed."""
    return _binary_available()


def supports(spec: dict[str, Any]) -> bool:
    """Whether this ruleset has a Fairy-Stockfish counterpart."""
    return _translates(spec)


class FairyStockfishAgent:
    """A Fairy-Stockfish process playing one ruleset.

    One process per agent, started when the agent is built and stopped when it is
    dropped. A UCI engine is a conversation with state, so sharing one between two
    seats would have each of them searching the other's position -- and a tournament
    builds an agent per seat per game precisely so that a player can hold something.
    """

    def __init__(self, engine: Engine, nodes: int = DEFAULT_NODES) -> None:
        self.engine = engine
        self.nodes = max(1, int(nodes))
        self.name = f"fairystockfish:nodes={self.nodes}"
        self.stats = SearchStats()
        try:
            self.variant = translate(engine.spec)
        except Untranslatable as error:
            raise FairyEngineError(
                f"Fairy-Stockfish cannot play {engine.spec.name!r}: {error}"
            ) from error
        self._uci = UciEngine()
        self._uci.load(self.variant, config_path(self.variant))

    def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
        """Ask the engine, and refuse anything our own engine does not offer.

        ``rng`` is accepted and unused, which is worth stating rather than hiding: this
        player's randomness is inside another process and no seed of ours reaches it.
        The signature is the `Agent` protocol's and the seat's stream is still drawn
        from by whoever else is in the game, so the *opponent's* choices are unaffected
        by this player being here -- which is the property `seat_rng` exists for.
        """
        del rng
        self.stats.moves += 1
        self.stats.nodes += self.nodes
        if len(moves) == 1:
            # No search happened, so none is counted -- the same rule `SearchStats`
            # applies to every other agent, and the reason `depth_sum` stays honest.
            self.stats.nodes -= self.nodes
            return moves[0]

        text = fairy_position.fen(
            self.engine.spec,
            state,
            self.variant.chars,
            self.variant.king,
            drops=self.variant.drops,
            check_limit=(
                self.engine.spec.check_limit if self.variant.check_counting else 0
            ),
            castling=self.variant.castling,
        )
        played = self._uci.best_move(text, self.nodes)
        if played is None:
            raise MoveMismatch(
                f"Fairy-Stockfish found no move in a position with {len(moves)} of "
                f"them: {text}"
            )
        return fairy_position.move_of(
            self.engine.spec,
            state,
            moves,
            played,
            self.variant.chars,
            self.variant.promotion_suffix,
        )

    def close(self) -> None:
        uci = getattr(self, "_uci", None)
        if uci is not None:
            uci.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        with contextlib.suppress(Exception):
            self.close()


@register("fairystockfish")
def _build_fairy_stockfish(engine: Engine, options: dict[str, Any]) -> FairyStockfishAgent:
    return FairyStockfishAgent(engine, nodes=int(options.get("nodes", DEFAULT_NODES)))


# Deliberately no `register_native`. The core has no idea this player exists, which is
# what sends every game it plays down `play_game_python` -- and that is the extension
# point working, not a fallback being tolerated.
