"""The learned evaluator's player name, its stores, and the integer oracle (Q502e).

Design record: ``docs/reference/learned-evaluator.md``.

**The name.** ``minimax:net=<procedure>,nodes=N`` (and ``greedy:net=<procedure>``) names a
*training procedure*, the same string on every board, as ``values=`` does; the net
it produced for a ruleset is a file, ``nets/<ruleset-hash>-<procedure>.json`` beside its
weights ``nets/<ruleset-hash>-<procedure>.net``. **Absent, refused or unreadable, the
player refuses to play** (Q709): every place a ``net=`` player is built -- the core's
parameters (``native_spec``, so the schedule, the practice player and the site's matches
alike) and the Python oracle -- resolves through ``require_net``, which raises
``NetError`` in a plain sentence rather than send the core no bytes. A player with no
bytes is ``Objective::Builtin`` in ``agents.rs``, so the old fallback played the built-in
evaluation under the learned player's name, and a person who chose the learned player got
*hard* without being told (`docs/reference/intransitive-quiet-endgame.md` §b.5).
``net_bytes`` keeps the permissive reading for the callers that ask *which* player a name
would be without seating it: a measurement's basis stamp (``builtin`` where a store is
missing) and the trainer's own parent check.

**The oracle.** ``NetEvaluator`` here is a declared test oracle under CLAUDE.md rule 3:
it computes the same integer the core does, from a full refresh every time, and
``tests/test_nets.py`` holds the two together bit for bit on random nets and positions,
and holds a Python search on it to the core's search move for move. Nothing outside
tests and the Python agent loop may use it to answer a question the core can answer.

No trainer here; that is Q502f, which writes the stores this module reads.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.compile import TIE_INITIATIVE

# Mirrors `rust/src/net.rs`; `tests/test_nets.py` holds the two together.
MAGIC = b"CVSNET01"
INPUTS_VERSION = 1
HAND_CAP = 16
CHECK_CAP = 8
POINTS_CAP = 16
CLOCK_BUCKETS = 8
TURN_CAP = 4
WEIGHT_SHIFT = 6
ACTIVATION_MAX = 127
OUTPUT_LIMIT = 1 << 27
OUTPUT_SCALE = 1024.0
MAX_ACCUMULATOR = 2048
MAX_HIDDEN = 128

STORE_DIR = "nets"
HASH_LENGTH = 12
ACCEPTED = "accepted"
REFUSED = "refused"


class NetError(ValueError):
    """A store that cannot be read as the net it claims to be."""


# ------------------------------------------------------------------ the input layout


@dataclass(frozen=True, slots=True)
class Layout:
    """Where each input segment starts, from the compiled spec's sizes alone."""

    n_squares: int
    n_values: int
    n_types: int
    unmoved: int
    ep: int
    colour: int
    hands: int
    checks: int
    points: int
    halfmove: int
    ply: int
    turn: int
    n_inputs: int
    no_progress: int
    max_plies: int
    # Q918g: a round game's own state after ``turn`` (``net.rs::Layout``), each segment only
    # where its rule is on, so every layout without it is unchanged: who holds the
    # initiative (2, under ``tie: "initiative"``), the void run's two squares and the barred
    # pairs' squares (2 x S each, in every round game). And (Q983b, `docs/plan-rest.md` §5.4)
    # the resting pieces, in every game that rests, alternating or in rounds: 2 x S x R
    # one-hot per (turns left, player, square), ``((left - 1) * 2 + player) * S + square``,
    # ``R`` the longest rest (``rest_turns``); for ``R = 1`` a round game's plane is schema
    # 27's, so its layout is unchanged.
    initiative: int = 0
    rest: int = 0
    rest_turns: int = 0
    void: int = 0
    bars: int = 0
    has_initiative: bool = False
    has_rest: bool = False
    rounds: bool = False

    @classmethod
    def of(cls, spec: Any) -> Layout:
        s, t = spec.n_squares, spec.n_types
        n_values = spec.n_owners * spec.n_facings * t
        unmoved = s * n_values
        ep = unmoved + s
        colour = ep + s
        hands = colour + 2 * s
        checks = hands + 2 * t * HAND_CAP
        points = checks + 2 * CHECK_CAP
        halfmove = points + 2 * POINTS_CAP
        ply = halfmove + CLOCK_BUCKETS
        turn = ply + CLOCK_BUCKETS
        sim = getattr(spec, "simultaneous", None)
        rounds = sim is not None
        has_initiative = sim is not None and sim.tie == TIE_INITIATIVE
        rest_turns = max(spec.rest) if spec.rest else 0
        has_rest = rest_turns > 0
        initiative = turn + TURN_CAP
        rest = initiative + (2 if has_initiative else 0)
        void = rest + 2 * s * rest_turns
        bars = void + (2 * s if rounds else 0)
        return cls(
            n_squares=s,
            n_values=n_values,
            n_types=t,
            unmoved=unmoved,
            ep=ep,
            colour=colour,
            hands=hands,
            checks=checks,
            points=points,
            halfmove=halfmove,
            ply=ply,
            turn=turn,
            n_inputs=bars + (2 * s if rounds else 0),
            no_progress=int(spec.no_progress_draw),
            max_plies=int(spec.max_plies),
            initiative=initiative,
            rest=rest,
            rest_turns=rest_turns,
            void=void,
            bars=bars,
            has_initiative=has_initiative,
            has_rest=has_rest,
            rounds=rounds,
        )

    def features(self, state: GameState) -> list[int]:
        """The active inputs of ``state``, ascending -- ``net.rs::Layout::features``."""
        out: list[int] = []
        s = self.n_squares
        for square in range(s):
            byte = state.board[square]
            if byte and byte <= self.n_values:
                out.append(square * self.n_values + byte - 1)
        out.extend(self.unmoved + sq for sq in range(s) if state.unmoved >> sq & 1)
        if 0 <= state.ep_target < s:
            out.append(self.ep + state.ep_target)
        for square, c in enumerate(state.colour[:s]):
            if c in (1, 2):
                out.append(self.colour + square * 2 + c - 1)
        for player in range(2):
            for piece_type, count in enumerate(state.hands[player][: self.n_types]):
                base = self.hands + (player * self.n_types + piece_type) * HAND_CAP
                out.extend(base + k for k in range(min(count, HAND_CAP)))
        for player in range(2):
            base = self.checks + player * CHECK_CAP
            out.extend(base + k for k in range(min(state.checks[player], CHECK_CAP)))
        for player in range(2):
            base = self.points + player * POINTS_CAP
            whole = min(math.floor(max(state.points[player], 0.0)), POINTS_CAP)
            out.extend(base + k for k in range(whole))
        out.extend(_clock(self.halfmove, state.halfmove, self.no_progress))
        out.extend(_clock(self.ply, state.ply, self.max_plies))
        extra = min(max(state.remaining - 1, 0), TURN_CAP)
        out.extend(self.turn + k for k in range(extra))
        if self.has_initiative and state.initiative in (0, 1):
            out.append(self.initiative + state.initiative)
        if self.has_rest:
            out.extend(sorted(
                self.rest + ((left - 1) * 2 + player) * s + square
                for player in range(2)
                for square, left in state.rest[player]
                if 0 <= square < s and 1 <= left <= self.rest_turns
            ))
        if not self.rounds:
            return out
        for player in range(2):
            if 0 <= state.void_run[player] < s:
                out.append(self.void + player * s + state.void_run[player])
        barred = {
            self.bars + player * s + bar[player]
            for player in range(2)
            for bar in state.bars
            if 0 <= bar[player] < s
        }
        out.extend(sorted(barred))
        return out


def _clock(base: int, value: int, limit: int) -> list[int]:
    if limit == 0:
        return []
    return [base + k for k in range(CLOCK_BUCKETS) if value * CLOCK_BUCKETS >= (k + 1) * limit]


# ------------------------------------------------------------------ the net


@dataclass(frozen=True, slots=True)
class Head:
    w1: tuple[int, ...]
    b1: tuple[int, ...]
    w2: tuple[int, ...]
    b2: tuple[int, ...]
    w3: tuple[int, ...]
    b3: int


@dataclass(frozen=True, slots=True)
class Net:
    """A parsed store: the layers ``net.rs::Net`` reads, as Python ints."""

    n_inputs: int
    accumulator: int
    hidden1: int
    hidden2: int
    w0: tuple[int, ...]
    b0: tuple[int, ...]
    heads: tuple[Head, Head]

    @classmethod
    def parse(cls, data: bytes) -> Net:
        if data[:8] != MAGIC:
            raise NetError("the net store does not start with the CVSNET01 magic")
        if len(data) < 28:
            raise NetError("the net store is truncated")
        version, n_inputs, acc, h1, h2 = struct.unpack_from("<5I", data, 8)
        if version != INPUTS_VERSION:
            raise NetError(f"the net store was written for input layout {version}")
        if not (1 <= acc <= MAX_ACCUMULATOR and 1 <= h1 <= MAX_HIDDEN and 1 <= h2 <= MAX_HIDDEN):
            raise NetError(f"the net's widths ({acc}, {h1}, {h2}) are out of range")
        at = 28

        def take(fmt: str, n: int) -> tuple[int, ...]:
            nonlocal at
            size = struct.calcsize(f"<{n}{fmt}")
            if len(data) - at < size:
                raise NetError("the net store is truncated")
            values = struct.unpack_from(f"<{n}{fmt}", data, at)
            at += size
            return values

        w0 = take("h", n_inputs * acc)
        b0 = take("h", acc)
        heads = []
        for _ in range(2):
            heads.append(
                Head(
                    w1=take("b", h1 * acc),
                    b1=take("i", h1),
                    w2=take("b", h2 * h1),
                    b2=take("i", h2),
                    w3=take("b", h2),
                    b3=take("i", 1)[0],
                )
            )
        if at != len(data):
            raise NetError(f"the net store has {len(data) - at} bytes after its last layer")
        return cls(n_inputs, acc, h1, h2, w0, b0, (heads[0], heads[1]))

    def accumulate(self, features: list[int]) -> list[int]:
        """A fresh ``i16`` accumulator: the wrapping sum of the bias and the rows."""
        acc = list(self.b0)
        width = self.accumulator
        for i in features:
            row = self.w0[i * width : (i + 1) * width]
            for j in range(width):
                acc[j] += row[j]
        return [((a + 32768) & 0xFFFF) - 32768 for a in acc]

    def forward(self, acc: list[int], side: int) -> int:
        """The raw output for the side to move -- ``net.rs::Net::forward``."""
        head = self.heads[side & 1]
        a0 = [min(max(a, 0), ACTIVATION_MAX) for a in acc]
        width = self.accumulator
        a1 = []
        for k in range(self.hidden1):
            row = head.w1[k * width : (k + 1) * width]
            total = head.b1[k] + sum(w * a for w, a in zip(row, a0, strict=True))
            a1.append(min(max(total >> WEIGHT_SHIFT, 0), ACTIVATION_MAX))
        out = head.b3
        for k in range(self.hidden2):
            row = head.w2[k * self.hidden1 : (k + 1) * self.hidden1]
            total = head.b2[k] + sum(w * a for w, a in zip(row, a1, strict=True))
            out += head.w3[k] * min(max(total >> WEIGHT_SHIFT, 0), ACTIVATION_MAX)
        return min(max(out, -OUTPUT_LIMIT), OUTPUT_LIMIT)


def encode(
    n_inputs: int,
    accumulator: int,
    hidden1: int,
    hidden2: int,
    w0: list[int],
    b0: list[int],
    heads: list[tuple[list[int], list[int], list[int], list[int], list[int], int]],
) -> bytes:
    """A store's bytes from its layers, in the order ``Net.parse`` reads them."""
    out = bytearray(MAGIC)
    out += struct.pack("<5I", INPUTS_VERSION, n_inputs, accumulator, hidden1, hidden2)
    out += struct.pack(f"<{len(w0)}h", *w0)
    out += struct.pack(f"<{len(b0)}h", *b0)
    for w1, b1, w2, b2, w3, b3 in heads:
        out += struct.pack(f"<{len(w1)}b", *w1)
        out += struct.pack(f"<{len(b1)}i", *b1)
        out += struct.pack(f"<{len(w2)}b", *w2)
        out += struct.pack(f"<{len(b2)}i", *b2)
        out += struct.pack(f"<{len(w3)}b", *w3)
        out += struct.pack("<i", b3)
    return bytes(out)


def random_net(
    n_inputs: int, seed: int, accumulator: int = 32, hidden1: int = 16, hidden2: int = 8
) -> bytes:
    """A net of random weights, for the tests that hold the core to the oracle.

    The ranges are chosen to exercise every clip: accumulator rows around a bias of
    about 40 so some slots go negative and some pass 127, and hidden biases that put
    sums on both sides of zero.
    """
    rng = random.Random(seed)
    w0 = [rng.randint(-24, 24) for _ in range(n_inputs * accumulator)]
    b0 = [rng.randint(-20, 100) for _ in range(accumulator)]
    heads = []
    for _ in range(2):
        heads.append(
            (
                [rng.randint(-128, 127) for _ in range(hidden1 * accumulator)],
                [rng.randint(-4000, 8000) for _ in range(hidden1)],
                [rng.randint(-128, 127) for _ in range(hidden2 * hidden1)],
                [rng.randint(-2000, 4000) for _ in range(hidden2)],
                [rng.randint(-128, 127) for _ in range(hidden2)],
                rng.randint(-5000, 5000),
            )
        )
    return encode(n_inputs, accumulator, hidden1, hidden2, w0, b0, heads)


class NetEvaluator:
    """The oracle objective: ``Objective::Net`` in ``features.rs``, refreshed every call.

    ``values`` is the derived table, read for capture ordering only, as the core's is.
    """

    def __init__(self, engine: Engine, net: bytes, values: tuple[float, ...]) -> None:
        self.net = Net.parse(net)
        self.layout = Layout.of(engine.spec)
        if self.net.n_inputs != self.layout.n_inputs:
            raise NetError(
                f"the net reads {self.net.n_inputs} inputs and this ruleset's layout has "
                f"{self.layout.n_inputs}; it was trained on another game and is refused"
            )
        self._values = values

    @property
    def values(self) -> tuple[float, ...]:
        return self._values

    def raw(self, state: GameState) -> int:
        acc = self.net.accumulate(self.layout.features(state))
        return self.net.forward(acc, state.side_to_move)

    def evaluate(
        self, engine: Engine, state: GameState, player: int, move_count: int | None = None
    ) -> float:
        score = self.raw(state) / OUTPUT_SCALE
        return score if player == state.side_to_move else -score

    def evaluate_round(
        self,
        engine: Engine,
        state: GameState,
        player: int,
        counts: tuple[int, int],
        lists: tuple[list[int], list[int]] | None = None,
    ) -> float:
        """Q918g: the net in a round, ``Objective::evaluate_round``'s net arm. A round
        game's side to move stays 0, so the one head read is seat 0's and seat 1's score
        is its negation; the move counts and lists are not read."""
        return self.evaluate(engine, state, player)


# ------------------------------------------------------------------ stores and names


@dataclass(frozen=True, slots=True)
class NetRecord:
    """What a store's JSON says about its net. The weights are the ``.net`` beside it."""

    procedure: str
    ruleset_hash: str
    verdict: str
    weights: bytes
    build: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.weights).hexdigest()[:HASH_LENGTH]

    @property
    def playing(self) -> bytes:
        """The bytes a player carries: the net if accepted, nothing if refused."""
        return self.weights if self.verdict == ACCEPTED else b""


_NETS: dict[tuple[str, str], NetRecord] = {}


def normalise_procedure(procedure: Any) -> str:
    """A procedure digest as it came back out of an agent string (leading zeros)."""
    return str(procedure).zfill(HASH_LENGTH)


def store_paths(
    ruleset_hash: str, procedure: str, root: str | Path = STORE_DIR
) -> tuple[Path, Path]:
    stem = Path(root) / f"{ruleset_hash}-{procedure}"
    return stem.with_suffix(".json"), stem.with_suffix(".net")


def register_net(record: NetRecord) -> None:
    _NETS[(record.procedure, record.ruleset_hash)] = record


def unregister_net(procedure: str, ruleset_hash: str) -> None:
    _NETS.pop((procedure, ruleset_hash), None)


def save_net(record: NetRecord, root: str | Path = STORE_DIR) -> Path:
    """Write a store: the JSON record and its weights. Q502f's trainer is the writer."""
    meta, weights = store_paths(record.ruleset_hash, record.procedure, root)
    meta.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(record.weights)
    Net.parse(record.weights)
    payload = {
        "procedure": record.procedure,
        "ruleset_hash": record.ruleset_hash,
        "inputs_version": INPUTS_VERSION,
        "verdict": record.verdict,
        "weights_sha256": hashlib.sha256(record.weights).hexdigest(),
        "build": record.build,
    }
    meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return meta


def load_net(
    ruleset_hash: str, procedure: str, root: str | Path = STORE_DIR
) -> NetRecord | None:
    """The stored net for a ruleset, or ``None`` when there is none (not an error)."""
    meta, weights = store_paths(ruleset_hash, procedure, root)
    if not meta.exists():
        return None
    data = json.loads(meta.read_text(encoding="utf-8"))
    if data.get("procedure") != procedure or data.get("ruleset_hash") != ruleset_hash:
        raise NetError(f"{meta} is filed under {ruleset_hash}/{procedure} and says otherwise")
    if data.get("inputs_version") != INPUTS_VERSION:
        raise NetError(f"{meta} was trained on input layout {data.get('inputs_version')}")
    blob = weights.read_bytes() if weights.exists() else b""
    if data.get("verdict") == ACCEPTED and hashlib.sha256(blob).hexdigest() != data.get(
        "weights_sha256"
    ):
        raise NetError(f"{weights} does not hash to what {meta} records")
    record = NetRecord(
        procedure=procedure,
        ruleset_hash=ruleset_hash,
        verdict=str(data.get("verdict")),
        weights=blob,
        build=str(data.get("build") or ""),
    )
    register_net(record)
    return record


def net_bytes(options: dict[str, Any], ruleset_hash: str) -> bytes:
    """The net an agent's ``net=`` option names on this ruleset, or ``b""``.

    ``b""`` for no ``net=``, for no store, and for a refused one. This is the reading for
    asking which player a name *would* be; a player that sits down resolves through
    ``require_net``, which refuses the last two. A ``net=`` player resolved without a
    ruleset raises, since the name is the same on every board and only the board says
    which net.
    """
    procedure = options.get("net")
    if procedure is None:
        return b""
    procedure = normalise_procedure(procedure)
    if not ruleset_hash:
        raise NetError(
            f"a net player needs to know which ruleset it is playing; procedure "
            f"{procedure!r} was resolved without one"
        )
    record = _NETS.get((procedure, ruleset_hash))
    if record is None:
        record = load_net(ruleset_hash, procedure)
    return b"" if record is None else record.playing


def require_net(options: dict[str, Any], ruleset_hash: str) -> bytes:
    """The bytes a ``net=`` player plays, or ``NetError`` where it has none (Q709).

    ``b""`` only for a player that names no net. A player that names one and finds no
    store, a refused or unchecked one, or one that cannot be read (a missing ``.net``,
    weights that do not hash to their record, a JSON that will not parse) refuses in a
    sentence saying which and why: it never plays the built-in evaluation under the
    learned player's name.
    """
    procedure = options.get("net")
    if procedure is None:
        return b""
    procedure = normalise_procedure(procedure)
    if not ruleset_hash:
        return net_bytes(options, ruleset_hash)  # raises: which ruleset is it playing?
    try:
        blob = net_bytes(options, ruleset_hash)
    except (OSError, ValueError) as error:
        why = f"its store cannot be read ({error})"
        raise NetError(_unplayable(procedure, ruleset_hash, why)) from error
    if blob:
        return blob
    record = _NETS.get((procedure, ruleset_hash))
    why = (
        "no net is stored for it" if record is None
        else f"its net was {record.verdict}, not accepted"
    )
    raise NetError(_unplayable(procedure, ruleset_hash, why))


def _confirmed(data: dict[str, Any]) -> bool:
    """Whether a store's JSON records a passed acceptance duel and a passed G-A2 on fresh seeds.

    A kept generation passed both by R5-1's definition (`archive/plan-teacher.md` §9.6), so the
    ``kept`` field stands for them where the second duel was recorded under another hash
    (*Intransitive*'s generations carried over by Q696)."""
    if data.get("kept") is True:
        return True
    duels = [data.get("acceptance"), data.get("confirmation")]
    return all(isinstance(duel, dict) and duel.get("accepted") is True for duel in duels)


def rated_net(ruleset_hash: str, root: str | Path = STORE_DIR) -> str:
    """The procedure of the net this ruleset's measuring roster plays, or ``""`` (Q502h).

    The user's answer of 2026-10-03 (`docs/archive/decisions.md`, *When a better computer
    player arrives*): each kept generation joins the measuring set at once. The rule, read
    off the stores alone so that a newly kept generation moves the basis the moment its
    JSON lands (`docs/reference/general-player.md` §7):

    - only an **accepted** store whose weights load and hash to their record plays;
    - it must be **kept** (R5-1) or, where no generation line runs, **accepted and
      confirmed** (its G-A2 duel on fresh seeds passed too, `archive/plan-teacher.md` §8.3); a
      store the plan read and did not keep (``kept: false``) never plays;
    - where any eligible store is kept, the choice is among the kept ones, so an accepted
      child stored beside its kept parent does not replace it;
    - of those, the newest generation (the one no other candidate was trained from), then
      the higher acceptance score, then the procedure, so the answer never depends on the
      order files are listed in.

    That is the site's learned player (`web.opponents._kept`) less any net not confirmed
    on a second seed set; a test holds the two together on every library game.
    """
    found: list[tuple[str, str, bool, float]] = []
    for path in sorted(Path(root).glob(f"{ruleset_hash}-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("verdict") != ACCEPTED:
            continue
        procedure = str(data.get("procedure") or "")
        if (data.get("ruleset_hash") != ruleset_hash or not procedure
                or path.name != f"{ruleset_hash}-{procedure}.json"
                or data.get("kept") is False or not _confirmed(data)):
            continue
        try:
            record = load_net(ruleset_hash, procedure, root)
        except (OSError, ValueError):
            continue
        if record is None or not record.playing:
            continue
        lineage = data.get("lineage")
        parent = str(lineage.get("parent") or "") if isinstance(lineage, dict) else ""
        acceptance = data.get("acceptance")
        score = float(acceptance.get("score", 0.0)) if isinstance(acceptance, dict) else 0.0
        found.append((procedure, parent, data.get("kept") is True, score))
    kept = [entry for entry in found if entry[2]]
    pool = kept or found
    parents = {entry[1] for entry in pool}
    leaves = [entry for entry in pool if entry[0] not in parents]
    if not leaves:
        return ""
    return max(leaves, key=lambda entry: (entry[3], entry[0]))[0]


def _unplayable(procedure: str, ruleset_hash: str, why: str) -> str:
    return (
        f"the learned player net={procedure} cannot play on ruleset {ruleset_hash}: {why}; "
        f"it refuses rather than play the built-in evaluation under the learned player's name"
    )


def played_by(agent: str, ruleset_hash: str) -> str:
    """Which net a seat really plays, for the record beside its agent string (Q709).

    ``net:<12 hex>`` (``NetRecord.digest``, the SHA-256 of the bytes the core is sent, as
    ``Basis.net`` stamps it) for a ``net=`` player, ``""`` for every other player, whose
    agent string already says everything about it. Strict as ``require_net`` is: a net
    player that cannot be loaded raises rather than be recorded as anything.
    """
    from chess_variant_sandbox.agents.base import parse_agent

    blob = require_net(parse_agent(agent).options, ruleset_hash)
    return "net:" + hashlib.sha256(blob).hexdigest()[:HASH_LENGTH] if blob else ""
