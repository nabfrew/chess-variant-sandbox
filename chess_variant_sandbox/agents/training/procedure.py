"""What a training procedure is, and the digest that names it in ``minimax:net=<digest>``.

A procedure is every choice that decides which net comes out, given a ruleset and a core:
who plays the self-play games and how many, which positions become samples and how they
are labelled, the network's widths, and the optimiser's schedule and seed. Its digest is
the ``net=`` name, the same string on every board, exactly as ``values=`` names a search
procedure (``learned-evaluator.md`` §1). Two procedures that differ in anything differ in
name.

The digest names the *recipe*; the net a recipe made on one machine is its weights, and
the basis stamps those (``Basis.net``). The recipe is deterministic on one machine and
core (``tests/test_training.py`` holds the store byte for byte), not across machines: a
float sum in another order is another net, which is why the file, not the recipe, is the
player (``general-player.md`` §5 Q4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from chess_variant_sandbox.agents.nets import (
    HASH_LENGTH,
    INPUTS_VERSION,
    MAX_ACCUMULATOR,
    MAX_HIDDEN,
)

# Bumped whenever the trainer's arithmetic changes what a procedure produces: the loss,
# the initialisation, the optimiser, the quantization, the sampling. It is in the digest,
# so a changed trainer never writes over a store an older one made.
TRAINER_VERSION = 1

# The acceptance duel (§2 E: "only if it beats the built-in objective by a pre-registered
# margin, as `fit=` is (48 games at 6,400)"). Paired seeds, so the error is measured
# (`tournament/duel.py`), and the bar is two of those errors above even, the bar Q502i's
# repairs were graded on. Constants of the verdict, not of the procedure: a verdict is
# about a net, and the net does not depend on how it will be judged.
#
# Q643 (learned-evaluator.md §8.11): the duel samples positions. Each of 200 pairs opens
# with up to `ACCEPT_OPENING_PLIES` random plies drawn exactly as self-play's openings are
# (a pure function of the pair's seed), on seeds from 1,000,000, which no self-play
# schedule of a plausible size reaches. Q502f2-Q637's duel (24 pairs from the start
# position, seed 0) is `opening_plies=0, pairs=24, seed=0`: on gardner it replayed about
# six positions.
ACCEPT_NODES = 6_400
ACCEPT_PAIRS = 200
ACCEPT_SEED = 1_000_000
ACCEPT_OPENING_PLIES = 8
ACCEPT_ERRORS = 2.0

# Q918g (`docs/archive/plan-simultaneous.md` §6.4, *Q918g, scoped*): a round game's acceptance duel is at
# the round roster's bottom rung, `matrix:nodes=1600` on both seats (§7: "beats
# `matrix:nodes=1600` there"), sixteen times `corpus.CANONICAL_AGENTS`' 100 as every round
# budget is (`loop/screen.ROUND_BUDGET_FACTOR`). The rest of the duel is `ACCEPT_*`'s: 200
# pairs from seed 1,000,000, up to 8 random plies (4 rounds), two errors above even.
ROUND_ACCEPT_NODES = 1_600

# The searches a label may come from (`Procedure.label_search`). "parent" (Q677) is the
# player's search on the parent net, `minimax:net=<parent>,nodes=<label_nodes>`.
LABEL_SEARCHES = ("probe", "player", "parent")

# Q918i (`docs/archive/plan-simultaneous.md` §7, *Q918h, the re-plan*, part 3): what a round teacher's
# matrix is solved on (`Procedure.round_payoff`). "units" is the evaluator's built-in units,
# an ending ±(10⁶ − ply), squashed once at the end (every recipe before it); "score" is each
# cell squashed first, an ending 1, ½ or 0, and the value used as the label as it is.
ROUND_PAYOFFS = ("units", "score")

# A generation (plan-teacher §9.1): the self-play budget of both seats, the games per
# ordered pairing of the two-player roster (four pairings, 20,000 games, half the seats
# the net's) and the parent's label budget, the strength it was accepted at.
GENERATION_SELF_PLAY_NODES = 1_600
GENERATION_GAMES = 5_000
GENERATION_LABEL_NODES = 6_400

_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Parent:
    """The net a generation learns from: its procedure's digest and its weights' SHA-256.

    The procedure alone does not name a net (the same recipe on another machine, or on
    another game, is other weights), so a generation's name carries the bytes' hash too:
    two parents that differ in one weight are two procedures. It is per game, as the
    weights are: a generation of *Intransitive* and one of gardner are two names.
    """

    procedure: str
    weights_sha256: str

    def __post_init__(self) -> None:
        if len(self.procedure) != HASH_LENGTH or not set(self.procedure) <= _HEX:
            raise ValueError(f"a parent's procedure is {HASH_LENGTH} hex, not {self.procedure!r}")
        if len(self.weights_sha256) != 64 or not set(self.weights_sha256) <= _HEX:
            raise ValueError("a parent's weights_sha256 is 64 hex (SHA-256 of its .net)")


def net_option(agent: str) -> str | None:
    """The ``net=`` value of an agent string, as ``nets.normalise_procedure`` reads it."""
    _kind, _, rest = agent.partition(":")
    for chunk in rest.split(","):
        key, _, value = chunk.partition("=")
        if key.strip() == "net":
            return value.strip().zfill(HASH_LENGTH)
    return None


@dataclass(frozen=True)
class Replay:
    """E1's replay set (plan-teacher §9.13.3): the parent's own training games, played again.

    A generation's new games narrowed to its parent's own play and to routs of the built-in
    (archive/plan-teacher §9.13.1), so the child also trains on the games its parent learned
    from: the parent's self-play schedule (its roster, games per pairing, seed and random
    openings), re-played
    on this ruleset and this core, and labelled by the child's own teacher (each arm's: the
    parent for the child, the probe for C). The positions are sampled as the procedure's
    (``skip_plies``, ``stride``).

    **The built-in-only block is left out by construction**: a game of the schedule in which
    no seat names a net is not played (``data.replay_tasks``). In a generation those games
    are the built-in player against itself, the same games as the new schedule's own block
    when the two schedules share their games, seed and built-in seat (E1's do), so they
    would only be counted twice.

    ``net`` is the net the replay roster names (the parent's parent, one generation back),
    by its procedure and its weights' SHA-256: it plays only as the bytes its name carries
    (``data.replay_bytes``), as the parent does. It is required, since a roster naming no
    net would leave nothing once its built-in-only games are left out.

    ``Replay.of(parent_recipe)`` is the one way E1 builds it: the parent's *new* games, never
    its own replay set, so the data stay about twice a generation's (§9.13.3).

    **The chain** (E3b, plan-teacher §9.17.4): ``prior`` is the replay set behind this one,
    the parent's own replay set, so a game's earliest generations' games never leave the
    union. ``links`` walks it: one generation back, then two back, and so on. Each link is a
    ``Replay`` of its own, so each roster may name only its own net. ``prior`` enters the
    digest only when set, so every one-link replay keeps its name. ``replayed_deep`` builds it.
    """

    agents: tuple[str, ...]
    games: int
    seed: int
    opening_plies: int
    net: Parent
    prior: Replay | None = None

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("a replay set needs the roster its games were played by")
        if self.games < 1:
            raise ValueError("a replay set's games per pairing must be at least 1")
        if self.opening_plies < 0:
            raise ValueError("a replay set's opening_plies must be 0 or more")
        named = {net_option(agent) for agent in self.agents} - {None}
        if not named:
            raise ValueError(
                "a replay roster naming no net is all built-in-only games, which a replay "
                "leaves out; there would be nothing to replay"
            )
        if named - {self.net.procedure}:
            raise ValueError(
                f"a replay roster may name only its own net (net={self.net.procedure}); it "
                f"names {', '.join(sorted(map(str, named)))}"
            )

    @classmethod
    def of(cls, parent_recipe: Procedure) -> Replay:
        """The parent's training games: its recipe's self-play, one generation back."""
        if parent_recipe.parent is None:
            raise ValueError(
                "the parent's recipe has no parent of its own, so its games name no net; a "
                "replay set is a generation's games (plan-teacher §9.13.3)"
            )
        return cls(
            agents=tuple(parent_recipe.agents),
            games=parent_recipe.games,
            seed=parent_recipe.seed,
            opening_plies=parent_recipe.opening_plies,
            net=parent_recipe.parent,
        )

    @property
    def links(self) -> tuple[Replay, ...]:
        """This replay set and every one behind it, nearest first (one back, two back, ...)."""
        out: list[Replay] = []
        link: Replay | None = self
        while link is not None:
            out.append(link)
            link = link.prior
        return tuple(out)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "agents": list(self.agents),
            "games": self.games,
            "seed": self.seed,
            "opening_plies": self.opening_plies,
            "net": asdict(self.net),
        }
        # E3b: absent unless the replay chains, so no one-link replay's name moves.
        if self.prior is not None:
            data["prior"] = self.prior.as_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Replay:
        prior = data.get("prior")
        return cls(
            agents=tuple(data["agents"]),
            games=int(data["games"]),
            seed=int(data["seed"]),
            opening_plies=int(data["opening_plies"]),
            net=Parent(**data["net"]),
            prior=cls.from_dict(prior) if prior is not None else None,
        )


@dataclass(frozen=True)
class Procedure:
    """One training recipe. The defaults are ``db5ef1bc938e`` (Q671, plan-teacher §3 rule 1):
    Q637's recipe (§8.10: Q502f2's with diverse self-play and the probe's label alone) at
    TD(λ 0.7), the first recipe whose net was accepted (Intransitive, §8.15-§8.16).
    ``Q637`` below is the default before it, ``Q502F2`` the first recipe."""

    # Self-play: the roster `tournament.runner` plays (every pairing, self-play included),
    # games per ordered pairing, and the schedule's seed.
    agents: tuple[str, ...] = ("minimax:nodes=1600",)
    games: int = 20_000
    seed: int = 0
    # Q637: diverse self-play. Each game opens with `k` plies of the core's `random` agent,
    # `k` uniform in `0..opening_plies` and every move's stream drawn from the game's
    # schedule seed (`data.opening_plies_for`, `data.opening_seed`), then the roster plays
    # it out from there. 0 is Q502f2's self-play, every game from the start position,
    # which on gardner made 2.8 % of the positions distinct (§8.9). Left out of the digest
    # at 0, so the recipes named before it existed keep their names.
    opening_plies: int = 8
    # Which plies become samples. Every game of a ruleset shares its first moves, whose
    # label is only the base rate, so a few are skipped; the terminal position is never a
    # sample (no move is chosen there, and its verdict is the core's, not the net's).
    skip_plies: int = 2
    stride: int = 1
    # The label: `result_weight` x the game's result for the side to move, plus the rest
    # x the probe's score squashed by `scale` (§2 E's λ, not `td_lambda`'s). `label_nodes` 0 is results only.
    # 0 since Q637: on Intransitive the result half was a coin flip that dragged the net's
    # agreement with the probe from 0.99 to 0.55 (§8.9); Q502f2's recipe had 0.5.
    result_weight: float = 0.0
    label_nodes: int = 1_200
    label_depth: int = 8
    # Q658 (plan-teacher §2.2): TD(λ) along the game line. The probe's part of the label
    # becomes `(1 - λ) x` this position's probe value `+ λ x` the next sampled position's
    # label of the same game (from this side's view), the last one ending in the result:
    # `data.td_labels`. 0 is the probe alone (every recipe before it), 1 the result alone.
    # Left out of the digest at 0, so no existing name moves. 0.7 since Q671: the λ 0.7
    # arm of Q637's recipe (`db5ef1bc938e`) was the first net accepted, on Intransitive
    # (D2 0.748 ± 0.019, confirmed on fresh seeds 0.741 ± 0.020; §8.15-§8.16).
    td_lambda: float = 0.7
    # Q665 (plan-teacher §8.2, T2v): which search labels a position. "probe" is the
    # analysis probe (`Engine.root_scores`, full width, at `label_nodes` and
    # `label_depth`), every recipe before it. "player" is the playing search's own root
    # value: `minimax:nodes=<label_nodes>` (alpha-beta, ordering, quiescence), the exact
    # value of its last completed iteration, its stream seeded from the game's seed and
    # the ply (`data.label_seed`); `label_depth` is not read. Left out of the digest at
    # "probe", so no existing name moves. "parent" (Q677) is the same search on the parent
    # net (`parent` below), `minimax:net=<parent>,nodes=<label_nodes>`.
    label_search: str = "probe"
    # Q918i: the round teacher's payoff (``ROUND_PAYOFFS``). "score" solves the one-round
    # matrix on expected score (`Core.round_values(..., score_scale=scale)`), so an ending
    # cell is worth 1 and not 10⁶ units; only the one-round teacher (`label_nodes` 1) of a
    # round recipe has it. Left out of the digest at "units", so no existing name moves.
    round_payoff: str = "units"
    # Built-in units (one mean non-royal piece) per logistic unit: the net's output `y`
    # predicts the side to move's expectation as `sigmoid(y / scale)`.
    scale: float = 1.0
    # The network (`net.rs` limits: A <= 2048, H1, H2 <= 128).
    accumulator: int = 256
    hidden1: int = 32
    hidden2: int = 32
    # The optimiser: Adam over shuffled minibatches, from `init_seed`.
    epochs: int = 8
    batch: int = 1_024
    learning_rate: float = 0.001
    init_seed: int = 0
    # Q677 (plan-teacher §9.1): a generation's parent, the net it learns from. Set, the
    # roster may name exactly this net (`net=<parent.procedure>`, no other) and
    # `label_search="parent"` labels with its player. Left out of the digest when unset,
    # so no existing name moves; set, the digest carries the parent's procedure and its
    # weights' SHA-256.
    parent: Parent | None = None
    # Q712 (plan-teacher §9.13.3, E1): a generation's replay set, the parent's own training
    # games played again beside the new ones and labelled by this procedure's teacher
    # (`Replay`). Left out of the digest when unset, so no existing name moves; set, the
    # digest carries the replay schedule and its net's weights' SHA-256. E3b (§9.17.4): it
    # may chain (`Replay.prior`, `replayed_deep`), every link replayed in order.
    replay: Replay | None = None

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("a training procedure needs at least one self-play agent")
        named = {net_option(agent) for agent in self.agents} - {None}
        if named and self.parent is None:
            # Generations (§2 E: each net generating the next net's data) are a roster
            # naming the previous net, and the same recipe only if the previous net is.
            # A string cannot carry the net's weights; `parent` does.
            raise ValueError(
                "a self-play roster naming a net is a generation; name the net's digest "
                "in the procedure (parent) before training from it"
            )
        if self.parent is not None and named - {self.parent.procedure}:
            raise ValueError(
                f"a generation's roster may name only its parent's net "
                f"(net={self.parent.procedure}); it names {', '.join(sorted(map(str, named)))}"
            )
        kinds = {agent.partition(":")[0].strip() for agent in self.agents}
        if "matrix" in kinds and kinds - {"matrix", "greedy", "random"}:
            raise ValueError(
                "a round game's roster (matrix:) plays rounds only; "
                f"{', '.join(sorted(kinds - {'matrix', 'greedy', 'random'}))} cannot sit beside it"
            )
        if self.rounds and self.searched and self.label_search == "probe":
            raise ValueError(
                "a round game has no probe teacher in the core; label it by the round "
                "player's own value (label_search 'player', Q918g)"
            )
        if self.rounds and (self.parent is not None or self.replay is not None):
            raise ValueError(
                "a round game's generations are not built (Q918g trains generation 0); "
                "train from self-play"
            )
        if self.round_payoff not in ROUND_PAYOFFS:
            raise ValueError(
                f"round_payoff {self.round_payoff!r} is not one of {', '.join(ROUND_PAYOFFS)}"
            )
        if self.round_payoff == "score" and not (
            self.rounds and self.searched and self.label_search == "player"
        ):
            raise ValueError(
                "a score payoff is a round teacher's (a matrix: roster, label_search 'player', "
                "result_weight < 1)"
            )
        if self.round_payoff == "score" and self.label_nodes != 1:
            raise ValueError(
                "a score payoff is the one-round teacher's (label_nodes 1, matrix:nodes=1); "
                "a deeper teacher's cells are sub-games solved in units"
            )
        if self.label_search == "parent" and self.parent is None:
            raise ValueError("label_search 'parent' needs a parent net (parent)")
        if self.replay is not None and self.parent is None:
            raise ValueError(
                "a replay set is a generation's (the parent's own training games); name the "
                "parent (parent) before replaying its games"
            )
        if not 1 <= self.accumulator <= MAX_ACCUMULATOR:
            raise ValueError(f"accumulator width {self.accumulator} is outside 1..{MAX_ACCUMULATOR}")
        for name in ("hidden1", "hidden2"):
            width = getattr(self, name)
            if not 1 <= width <= MAX_HIDDEN:
                raise ValueError(f"{name} width {width} is outside 1..{MAX_HIDDEN}")
        if not 0.0 <= self.result_weight <= 1.0:
            raise ValueError(f"result_weight {self.result_weight} is outside [0, 1]")
        if self.result_weight < 1.0 and self.label_nodes <= 0:
            raise ValueError("a search label needs label_nodes > 0, or result_weight 1")
        if not 0.0 <= self.td_lambda <= 1.0:
            raise ValueError(f"td_lambda {self.td_lambda} is outside [0, 1]")
        if self.td_lambda and not self.searched:
            raise ValueError(
                "td_lambda blends the probe's values; it needs result_weight < 1 "
                "(a results-only label takes td_lambda 0)"
            )
        if self.label_search not in LABEL_SEARCHES:
            raise ValueError(
                f"label_search {self.label_search!r} is not one of {', '.join(LABEL_SEARCHES)}"
            )
        for name in ("games", "stride", "epochs", "batch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.opening_plies < 0:
            raise ValueError("opening_plies must be 0 or more")
        if self.skip_plies < 0 or self.scale <= 0 or self.learning_rate <= 0:
            raise ValueError("skip_plies, scale and learning_rate must be positive")

    @property
    def rounds(self) -> bool:
        """Q918g: whether this recipe plays a round game -- its roster is the round
        players' (`matrix:`), and its player teacher is `matrix:nodes=<label_nodes>`. A
        roster cannot mix the two kinds: no game is played by both."""
        return any(agent.partition(":")[0].strip() == "matrix" for agent in self.agents)

    @property
    def searcher(self) -> str:
        """The searching player's kind on this recipe's game: ``matrix`` in rounds."""
        return "matrix" if self.rounds else "minimax"

    @property
    def searched(self) -> bool:
        """Whether positions are labelled by the probe as well as by the result."""
        return self.result_weight < 1.0

    @property
    def teacher(self) -> str:
        """Which search made a harvest's values (``Dataset.teacher``): the label search,
        and for the parent's player which parent, so one parent's values never train
        under another's name."""
        if self.label_search == "parent":
            assert self.parent is not None
            return f"parent:{self.parent.weights_sha256[:HASH_LENGTH]}"
        if self.round_payoff == "score":
            # Q918i: a score label is not a units label squashed; neither trains as the other.
            return f"{self.label_search}:score"
        return self.label_search

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agents"] = list(self.agents)
        # Absent at 0, so every recipe named before Q637 keeps its digest (`1c193cc56c76`
        # is still Q502f2's): a field that changes nothing must not rename anything.
        if not self.opening_plies:
            del data["opening_plies"]
        # The same for Q658's TD(λ): absent at 0, so `1d160257b8e5` and `1c193cc56c76` hold.
        if not self.td_lambda:
            del data["td_lambda"]
        # And for Q665's teacher: absent while it is the probe.
        if self.label_search == "probe":
            del data["label_search"]
        # And for Q918i's round payoff: absent at "units".
        if self.round_payoff == "units":
            del data["round_payoff"]
        # And for Q677's parent: absent unless this is a generation.
        if self.parent is None:
            del data["parent"]
        # And for Q712's replay set: absent unless E1's field is set.
        if self.replay is None:
            del data["replay"]
        else:
            data["replay"] = self.replay.as_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Procedure:
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        if "agents" in values:
            values["agents"] = tuple(values["agents"])
        if values.get("parent") is not None:
            values["parent"] = Parent(**values["parent"])
        if values.get("replay") is not None:
            values["replay"] = Replay.from_dict(values["replay"])
        # A recipe without the key was written before it existed, or at 0: either way 0,
        # never today's default.
        values.setdefault("opening_plies", 0)
        values.setdefault("td_lambda", 0.0)
        values.setdefault("label_search", "probe")
        values.setdefault("round_payoff", "units")
        return cls(**values)

    def with_(self, **changes: Any) -> Procedure:
        return replace(self, **changes)

    @property
    def digest(self) -> str:
        """The ``net=`` name: 12 hex over the recipe, the input layout and the trainer."""
        blob = json.dumps(
            {
                "procedure": self.as_dict(),
                "inputs_version": INPUTS_VERSION,
                "trainer_version": TRAINER_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:HASH_LENGTH]


DEFAULT = Procedure()
"""The default recipe (`db5ef1bc938e`, Q671): Q637's at TD(λ 0.7); its Intransitive net is accepted."""

ROUND_DEFAULT = Procedure(
    agents=("matrix:nodes=1600",), label_search="player", label_nodes=6_400
)
"""Q918g: the round games' generation 0 (`docs/archive/plan-simultaneous.md` §7, *Q918g, scoped*). The
default's recipe with the round players: self-play by `matrix:nodes=1600` (the round roster's
bottom rung), each position labelled by `matrix:nodes=6400`'s own value of the round for player
0 (the player teacher, four times the self-play budget as a generation's is; the core has no
probe for rounds), TD(λ 0.7) along the line, and everything else the default's."""

ROUND_CONTROL = ROUND_DEFAULT.with_(label_nodes=1)
"""Q918g's control C: the same games and positions, labelled by the one-round matrix alone
(`matrix:nodes=1` finishes pass 1 and nothing more, so its value is greedy's one-round matrix
game, the round analysis probe's reading at the players' ruler)."""

ROUND_SCORE = ROUND_CONTROL.with_(round_payoff="score")
"""Q918i's arm S (`docs/archive/plan-simultaneous.md` §7, *Q918h, the re-plan*, part 3): ``ROUND_CONTROL``
with one change, the one-round teacher's matrix solved on expected score (each cell player 0's
`sigmoid(units / scale)`, or 1, ½ or 0 where the round ends the game), its value the label."""

ROUND_OUTCOME = ROUND_DEFAULT.with_(result_weight=1.0, td_lambda=0.0, label_nodes=0)
"""Q918i's arm O: results only (`result_weight` 1, `td_lambda` 0, `label_nodes` 0), trained from
Q918g's saved positions; ``train --result-weight 1 --td-lambda 0 --label-nodes 0`` on a round game."""

Q637 = Procedure(td_lambda=0.0)
"""The default before Q671 (`1d160257b8e5`, §8.10): diverse self-play, the probe's label alone."""

Q502F2 = Procedure(opening_plies=0, result_weight=0.5, td_lambda=0.0)
"""The first real recipe (`1c193cc56c76`, §8.7): every game from the start, half the label the result."""


def generation(parent: Parent, base: Procedure = DEFAULT) -> Procedure:
    """Generation k + 1 of ``parent`` (plan-teacher §9.1): ``base``'s recipe with the parent
    net put where the built-in evaluator was.

    The roster is the parent's player and the built-in one, both at
    ``GENERATION_SELF_PLAY_NODES``, every ordered pairing (self-play included) at
    ``GENERATION_GAMES`` games, so half the seats are the net's; the label is the parent's
    player at ``GENERATION_LABEL_NODES`` (``label_search="parent"``), TD(λ) and the rest
    as ``base``. The student is trained from scratch (``init_seed`` as ``base``'s).
    """
    nodes = GENERATION_SELF_PLAY_NODES
    return base.with_(
        parent=parent,
        agents=(f"minimax:net={parent.procedure},nodes={nodes}", f"minimax:nodes={nodes}"),
        games=GENERATION_GAMES,
        label_search="parent",
        label_nodes=GENERATION_LABEL_NODES,
    )


def control(child: Procedure, base: Procedure = DEFAULT) -> Procedure:
    """§9.1's control C: ``base``'s teacher (the probe at its nodes) on the child's games.

    On a round game's recipe (Q918g) it is ``ROUND_CONTROL``'s teacher, the one-round matrix
    (``matrix:nodes=1``), since the core has no probe for rounds."""
    if child.rounds:
        return child.with_(label_search="player", label_nodes=ROUND_CONTROL.label_nodes)
    return child.with_(
        label_search=base.label_search, label_nodes=base.label_nodes, label_depth=base.label_depth
    )


def builtin_player_arm(child: Procedure) -> Procedure:
    """§9.1's arm K: the built-in player's own value at the child's label budget, on its games."""
    if child.rounds:
        raise ValueError(
            "arm K is a generation's (the built-in player beside the parent's); a round "
            "game's recipe is generation 0, whose teacher is already the built-in player"
        )
    return child.with_(label_search="player")


def replayed(child: Procedure, parent_recipe: Procedure) -> Procedure:
    """E1's child (plan-teacher §9.13.3, generation 2R): ``child`` with its parent's own
    training games as a replay set (``Replay.of(parent_recipe)``).

    ``parent_recipe`` is the recipe the parent was trained by (its store's ``recipe``), and
    its net must be ``child``'s parent: the store's procedure is the parent's name.
    ``control(replayed(...))`` is E1's C, the same union under the default's teacher.
    """
    if child.parent is None:
        raise ValueError("a replay set is a generation's; the child names no parent")
    if parent_recipe.digest != child.parent.procedure:
        raise ValueError(
            f"the recipe given is {parent_recipe.digest}, not the parent "
            f"{child.parent.procedure}'s: a replay set is the parent's own games"
        )
    return child.with_(replay=Replay.of(parent_recipe))


def replayed_deep(child: Procedure, parent_recipe: Procedure) -> Procedure:
    """E3b's chained child (plan-teacher §9.17.4): ``child`` with the parent's new games as
    its replay set and, behind them, the parent's own replay set, whole (so recursively
    every link the parent carried).

    On gardner, the child of ``5fe1cb87d4c9`` replays generation 1's games (the parent's
    new games, ``e03365fe43e7`` playing) and then generation 0's (the parent's replay set,
    ``db5ef1bc938e`` playing). A parent with no replay set gives ``replayed``'s child.
    """
    one = replayed(child, parent_recipe)
    assert one.replay is not None
    return one.with_(replay=replace(one.replay, prior=parent_recipe.replay))


# The fields that decide which games are played and which positions are sampled: two
# procedures equal on these harvest the same positions from one self-play (§9.1: the
# child, C and K share every position byte for byte; E1's replay set is part of it).
SELF_PLAY_FIELDS = (
    "agents", "games", "seed", "opening_plies", "skip_plies", "stride", "parent", "replay"
)


def same_positions(first: Procedure, second: Procedure) -> bool:
    return all(getattr(first, name) == getattr(second, name) for name in SELF_PLAY_FIELDS)
