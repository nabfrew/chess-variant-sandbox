"""A hint about a saved game's description: the sentences its rules seem not to support.

Q396, and optional by design. The user, 2026-09-24: *"Use Jev to verify description: B add
a hint. However, build it as optional nice-to-have, so that nothing relies on it, so it's
easy to roll back if it gets expensive."* The probe behind it is
`docs/reference/jev-probe.md` §3, which found the whole-description question too coarse
and said how to ask it instead: claim by claim, with the counts in the state.

**What it does.** Once a save has answered, the design page asks `POST /api/design/hint`
with the saved hash. This module reads the saved file, splits its description into
sentences, and asks TypeSafe's Jev one question per sentence in a single request: does
the sentence agree with the rules, contradict them, or say something the rules could not
confirm either way, such as history or opinion. The rules go in as `spec/describe.py`
renders them, with the name and the description taken out. Beside them go facts read off
the compiled spec: how many of each piece each side starts with, what is royal, and who
may take whom. Jev is weakest at counting, and `describe_spec` shortens a long setup to
"... (24 squares)", so the counts have to arrive already counted. The answer names the
sentences Jev reads as contradicted. A person reads it, and nothing else does.

This is a reading of a description against rule facts. It computes no legal move, no
outcome and no move choice (architecture rule 3), and nothing under `metrics/`, the
gates or the loop reads it.

**What nothing relies on.**
- *No key, no hint.* With `CHESS_VARIANT_SANDBOX_JEV_KEY` unset, `for_saved` answers `{"off": true}`
  before it reads a file, and nothing is sent anywhere. The page then stops asking.
- *The save never waits on it.* The save's route does not import this module and its
  answer does not change. The page asks here only after the save has answered and been
  drawn, and drops a failure of this request.
- *A failure is no hint.* A service error, a refusal, a malformed answer or a timeout
  (`TIMEOUT`) answers `{}`, never an error.
- *Past the day's cap, no hint.* `CHESS_VARIANT_SANDBOX_JEV_DAILY_CAP` (default `DEFAULT_CAP`) bounds
  the requests sent per UTC day. A value that is not a whole number is read as 0, so a
  typo switches the hint off rather than lifting the cap.

**What it costs, and where that is written.** One request per saved description, of up
to `MAX_CLAIMS` questions. `runs/jev-hint/calls.json` holds one row per UTC day: the
requests sent, the ones that failed, the asks the cap refused, the input tokens and what
they cost at `PRICE_PER_TOKEN`. Each answer is kept as `runs/jev-hint/<hash>.json`, and
asking again about the same saved description is answered from that file and costs
nothing. Jev does not return the same probabilities twice for one request (§4 of the
probe), so the stored observation is the one to show rather than a fresh one.

**Switching it off** is unsetting the key; nothing else changes. **Taking it out** is
deleting this module and the lines that name it: its call in `web/server.py`'s
`App.post`, its row in `site/routes.py` (and `MEMBERS` in `tests/test_site.py`), the
paragraph `site/pages.privacy` asks it for, and `askHint` in `web/static/page/design.js`.
`tests/test_jev_hint.py` goes with it.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chess_variant_sandbox.engine import CompiledSpec

KEY_VAR = "CHESS_VARIANT_SANDBOX_JEV_KEY"
CAP_VAR = "CHESS_VARIANT_SANDBOX_JEV_DAILY_CAP"

# Requests per UTC day. A request is one saved description, so this is a hundred new
# descriptions a day, more than the site sees. Q396's live check measured 1,700 to 2,700
# input tokens a request, about $0.0001, so a full day's cap is about a cent.
DEFAULT_CAP = 100

URL = "https://api.typesafe.ai/v1/systemone"
# Pinned rather than `jev-latest`, so a later model is a different stored answer and not
# a silent change under the same one. The probe pinned the same version.
MODEL = "jev-1.13.0"
# Input only; output is free (docs.typesafe.ai/models, 2026-09-24).
PRICE_PER_TOKEN = 0.042 / 1e6

# Seconds before the request is given up and the answer is no hint. Jev answers in about
# 0.7 s; the page draws nothing while it waits, and the save it follows has long answered.
TIMEOUT = 10.0

# The service sits behind Cloudflare, which refuses urllib's default user agent on some
# services (`site/mail.py` met it at Resend).
USER_AGENT = "chess-variant-sandbox/1"

# How many sentences one request asks about, and how much of each is sent. A description
# longer than this is read from its start.
MAX_CLAIMS = 12
MAX_CLAIM_CHARS = 300
# A sentence shorter than this many words is a fragment, not a claim.
MIN_WORDS = 3

# A sentence is named in the hint when Jev picks *contradicts* with at least this much
# probability. In Q396's live check the true sentences of three library descriptions
# read at most 0.26, the planted false ones 1.00, and one loosely worded true one 0.69
# (gardner's "the full chess army minus one bishop-pair partner and the second rook
# file"). A hint that can be wrong says so on the page.
THRESHOLD = 0.5

# Bumped when the question's wording changes, so a stored answer to the old question is
# not shown as an answer to the new one. What `rules` and `facts` say is part of the
# question: 2 is Q426, where a piece with no capturing move stopped being told it takes
# things, an unlimited stock stopped reading 1, and a stone nothing may take stopped
# being said to stay "until it is taken". An answer to 1 about such a game read a true
# sentence as contradicted.
VERSION = 2

HINT_DIR = "jev-hint"
CALLS_FILE = "calls.json"
# Days of the call count kept in `calls.json`.
KEEP_DAYS = 90

_DIGEST = re.compile(r"^[0-9a-f]{6,64}$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_CLAUSE_END = re.compile(r";\s+")

# One lock for the calls file, held only while it is read and rewritten. It is what makes
# the cap exact across the request threads of one server.
_LOCK = threading.Lock()


# ----------------------------------------------------------------- the two settings


def key() -> str:
    """The service key, or the empty string. Never printed, never stored."""
    return os.environ.get(KEY_VAR, "").strip()


def enabled() -> bool:
    """Whether the hint is switched on: a key is configured."""
    return bool(key())


def daily_cap() -> int:
    """Requests allowed per UTC day. Not a whole number, or below zero, reads as 0."""
    stated = os.environ.get(CAP_VAR, "").strip()
    if not stated:
        return DEFAULT_CAP
    try:
        return max(0, int(stated))
    except ValueError:
        return 0


# ------------------------------------------------------------------- the one entry


def for_saved(runs_dir: Path | str, digest: str) -> dict[str, Any]:
    """The hint for one saved design, or an answer that draws nothing.

    `{"off": true}` when no key is configured. `{}` when there is nothing to say or it
    could not be said: no such saved design, no description, the cap reached, or the
    service failed. Otherwise `claims` lists the sentences Jev reads as contradicted,
    which is empty when it found none, and `checked` says how many it read.
    """
    if not enabled():
        return {"off": True}
    try:
        return _hint(Path(runs_dir), digest)
    except Exception as error:  # a hint is never an error, whatever went wrong
        _say(f"no hint for {digest[:12]}: {type(error).__name__}")
        return {}


def _hint(runs_dir: Path, digest: str) -> dict[str, Any]:
    if not _DIGEST.match(digest):
        return {}
    saved = runs_dir / "designed" / "specs" / f"{digest}.json"
    if not saved.is_file():
        return {}
    spec = json.loads(saved.read_text(encoding="utf-8"))
    description = str(spec.get("description") or "").strip()
    found = claims(description)
    if not found:
        return {}

    stored_at = runs_dir / HINT_DIR / f"{digest}.json"
    stored = _stored(stored_at, description)
    if stored is None:
        if not _reserve(runs_dir):
            return {}
        state = {"rules": rules_text(spec), "facts": facts(spec), "claims": found}
        try:
            answer = _send(_request(state, len(found)), key())
        except Exception as error:
            _account(runs_dir, failed=True)
            _say(f"no hint for {digest[:12]}: the service did not answer "
                 f"({type(error).__name__})")
            return {}
        tokens = int((answer.get("usage") or {}).get("input_tokens") or 0)
        stored = _record(answer, found, description, digest, tokens)
        _account(runs_dir, tokens=tokens)
        stored_at.parent.mkdir(parents=True, exist_ok=True)
        stored_at.write_text(json.dumps(stored, indent=1), encoding="utf-8")
    return {
        "hash": digest,
        "checked": len(stored["claims"]),
        "claims": [
            {"claim": one["claim"], "contradicts": one["contradicts"]}
            for one in stored["claims"]
            if one["flagged"]
        ],
    }


# ------------------------------------------------------------------ the question


def claims(description: str) -> list[str]:
    """The description as sentences, and a sentence with a semicolon as its clauses.

    Split by code rather than asked for, because the question is about one claim at a
    time (§3 of the probe). Fragments under `MIN_WORDS` are dropped; the first
    `MAX_CLAIMS` are kept.
    """
    text = " ".join(description.split())
    found = []
    for sentence in _SENTENCE_END.split(text):
        for clause in _CLAUSE_END.split(sentence):
            clause = clause.strip()
            if len(clause.split()) >= MIN_WORDS:
                found.append(clause[:MAX_CLAIM_CHARS])
    return found[:MAX_CLAIMS]


def rules_text(spec: dict[str, Any]) -> str:
    """The rules as the library prints them, with the name and the description taken out.

    The description would otherwise answer for itself, and a name like *shogi* lets world
    knowledge vouch for rules that are not these.
    """
    from chess_variant_sandbox.spec.describe import describe_spec

    blind = dict(spec, name="Game", description="")
    blind.pop("provenance", None)
    return describe_spec(blind)


def facts(spec: dict[str, Any]) -> dict[str, Any]:
    """What a description is most often wrong about, counted off the compiled spec.

    The opening board, royalty and the capture matrix are the compiler's tables, so the
    rule is read where it is interpreted (architecture rule 4) and nothing here works
    out a square or a move.
    """
    from chess_variant_sandbox.agents.evaluation import inexhaustible_types
    from chess_variant_sandbox.engine import compile_spec

    compiled = compile_spec(spec)
    names = _names(compiled.piece_ids, compiled.piece_names)
    armies: list[Counter[int]] = [Counter(), Counter(), Counter()]
    for value in compiled.initial_board:
        if value:
            armies[compiled.owner_of(value)][compiled.type_of(value)] += 1

    def army(counts: Counter[int]) -> dict[str, Any]:
        return {"each kind": {names[kind]: counts[kind] for kind in sorted(counts)},
                "in all": sum(counts.values())}

    # A hexagon's absent corners (schema 16) are folded into `obstacles` for the tables;
    # they are not walls, and not cells either, so they are counted out of both.
    walls = len(compiled.obstacles - compiled.absent)
    cells = compiled.width * compiled.height - len(compiled.obstacles)
    if compiled.absent:
        board = (f"a regular hexagon of {cells} cells, {(compiled.width + 1) // 2} on a "
                 "side; every cell touches six others")
    elif compiled.lattice == "hex":
        board = f"hexagonal: {cells} cells on a {compiled.width} by {compiled.height} hex grid"
    else:
        board = (f"{compiled.width} files by {compiled.height} ranks, "
                 f"{compiled.width * compiled.height} squares")
    if walls:
        board += f", {walls} of them walled off so nothing may stand there"
    found: dict[str, Any] = {
        "board": board,
        "first player's army at the start (White, moves first)": army(armies[0]),
        "second player's army at the start (Black)": army(armies[1]),
        "pieces on the board at the start, both sides together": sum(
            sum(counts.values()) for counts in armies),
    }
    if armies[2]:
        found["pieces neither player owns, at the start"] = army(armies[2])
    stock = compiled.drops.stock if compiled.drops is not None else ()
    if any(stock):
        # An unlimited slot is stored as a constant 1 (`compile.DropRule`), and a 1 in
        # the facts told Jev a supply the description calls unlimited is one piece (Q132).
        endless = inexhaustible_types(compiled)
        found["held in hand by each player at the start"] = {
            names[kind]: "unlimited" if kind in endless else held
            for kind, held in enumerate(stock) if held}
    royal = [names[kind] for kind in compiled.royal_types]
    found["royal pieces (losing them loses the game)"] = royal or "none: no piece is royal"
    # Said from both sides, because the rules say it of the player it happens to and a
    # description says it of the player who does it. The live check's one false reading
    # was that step: "stalemating the opponent wins" against "a player with no legal move
    # loses" read as a contradiction at 0.98.
    found["a player left with no legal move"] = _BOTH_SIDES[compiled.stalemate].format(
        act="leaving the opponent with no legal move (stalemating them)")
    if royal:
        found["a player who is checkmated"] = _BOTH_SIDES[compiled.checkmate].format(
            act="checkmating the opponent")
    found["which enemy pieces each piece may take"] = _takes(compiled, names)
    return found


# What the capture fact says of a type that cannot capture at all, whatever the capture
# matrix permits it.
NO_CAPTURE = "none: it has no capturing move"


def _takes(compiled: CompiledSpec, names: list[str]) -> dict[str, Any] | str:
    """Who may take whom, said only of the types that can capture at all.

    ``capture_allowed`` is a permission, and a piece with no capturing move has every
    permission and uses none: a stone with no ``captures`` list was listed as taking
    everything, and Jev, told to trust the facts, read *"a stone is never captured"* as
    contradicted by it (Q132). So a type that cannot capture (`describe.capturing_types`,
    the same reading the rules text uses) says so, and nothing else is said of it.
    """
    from chess_variant_sandbox.spec.describe import capturing_types

    capturing = capturing_types(compiled)
    if not compiled.has_capture_matrix and all(capturing):
        return "every piece may take any kind of enemy piece; how each captures is in `rules`"
    return {
        names[attacker]: (
            [names[victim] for victim, allowed in enumerate(row) if allowed] or "none"
            if capturing[attacker]
            else NO_CAPTURE
        )
        for attacker, row in enumerate(compiled.capture_allowed)
    }


_BOTH_SIDES = {
    "draw": "draws, so {act} draws",
    "loss": "loses, so {act} wins",
    "win": "wins, so {act} loses",
}


def _names(ids: tuple[str, ...], named: tuple[str, ...]) -> list[str]:
    """Each piece type as a reader would call it, told apart where two share a name."""
    plain = [str(name or ident) for ident, name in zip(ids, named, strict=True)]
    shared = {name for name, count in Counter(plain).items() if count > 1}
    return [f"{name} ({ident})" if name in shared else name
            for ident, name in zip(ids, plain, strict=True)]


INSTRUCTIONS = (
    "`claims[{index}]` is one sentence from the description a designer wrote of their "
    "board game. `rules` are that game's complete rules, and `facts` are exact counts and "
    "relations computed from them; trust `facts` for any number. Is what the sentence "
    "says about the game true of these rules? Only its statements about the board, the "
    "pieces and how they move and capture, how many there are and where they start, and "
    "how the game is won, lost or drawn can be checked. History, credit, opinion, "
    "purpose, comparisons with other games and predictions about how play goes cannot."
)

CRITERIA = {
    "agrees": "The rules or the facts state what the sentence says about the game, or "
              "directly imply it.",
    "contradicts": "The sentence gets something checkable wrong: a board, a number or kind "
                   "of piece, a move or capture, a setup or a way of winning that the rules "
                   "or the facts show to be different, or a rule it relies on that these "
                   "rules do not have.",
    "not_checkable": "The sentence is history, credit, opinion, purpose, a comparison with "
                     "another game or a prediction about play, which the rules neither "
                     "confirm nor contradict.",
}


def _request(state: dict[str, Any], count: int) -> dict[str, Any]:
    return {
        "model": MODEL,
        "state": state,
        "questions": {
            f"claim_{index}": {
                "type": "choice",
                "instructions": INSTRUCTIONS.format(index=index),
                "criteria": CRITERIA,
            }
            for index in range(count)
        },
    }


def _record(
    answer: dict[str, Any], found: list[str], description: str, digest: str, tokens: int
) -> dict[str, Any]:
    """The answer as it is stored: every sentence, its choice and its probabilities."""
    answers = answer["answers"]
    rows = []
    for index, claim in enumerate(found):
        one = answers[f"claim_{index}"]
        probabilities = {str(k): float(v) for k, v in (one.get("probabilities") or {}).items()}
        contradicts = round(probabilities.get("contradicts", 0.0), 3)
        rows.append({
            "claim": claim,
            "choice": str(one.get("choice", "")),
            "probabilities": probabilities,
            "contradicts": contradicts,
            "flagged": one.get("choice") == "contradicts" and contradicts >= THRESHOLD,
        })
    return {
        "hash": digest,
        "description": description,
        "model": str(answer.get("model") or MODEL),
        "asked": MODEL,
        "version": VERSION,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokens": tokens,
        "claims": rows,
    }


def _stored(path: Path, description: str) -> dict[str, Any] | None:
    """The kept answer for this description under this question, or `None`."""
    if not path.is_file():
        return None
    try:
        kept = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    same = (kept.get("description") == description and kept.get("asked") == MODEL
            and kept.get("version") == VERSION)
    return dict(kept) if same else None


# ------------------------------------------------------------------ the wire


def _send(request: dict[str, Any], secret: str) -> dict[str, Any]:
    """One POST to the service, and its JSON. Raises on anything but a 200.

    `urllib` is imported here, so nothing reaches for a client until a key is set and a
    request is due. No retry: a failure is no hint, and the page is not waiting on it.
    """
    import urllib.request

    body = json.dumps(request).encode("utf-8")
    sent = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    })
    with urllib.request.urlopen(sent, timeout=TIMEOUT) as response:
        answer = json.load(response)
    if not isinstance(answer, dict) or not isinstance(answer.get("answers"), dict):
        raise ValueError("the answer has no answers")
    return answer


# ------------------------------------------------------------------ the count


def calls(runs_dir: Path | str) -> dict[str, dict[str, Any]]:
    """What `calls.json` says, by UTC day. Empty when nothing has been asked."""
    path = Path(runs_dir) / HINT_DIR / CALLS_FILE
    try:
        read = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return read if isinstance(read, dict) else {}


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _write(runs_dir: Path, days: dict[str, dict[str, Any]]) -> None:
    kept = dict(sorted(days.items())[-KEEP_DAYS:])
    path = runs_dir / HINT_DIR / CALLS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(kept, indent=1), encoding="utf-8")
    os.replace(partial, path)


def _day(days: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return days.setdefault(
        _today(), {"calls": 0, "failed": 0, "capped": 0, "tokens": 0, "dollars": 0.0})


def _reserve(runs_dir: Path) -> bool:
    """Count one request against today's cap before it is sent, or refuse it.

    Counted first so two threads cannot both take the last one. A count that cannot be
    written is a cap that cannot be kept, so that refuses too.
    """
    cap = daily_cap()
    try:
        with _LOCK:
            days = calls(runs_dir)
            day = _day(days)
            if day["calls"] >= cap:
                day["capped"] += 1
                _write(runs_dir, days)
                refused = True
            else:
                day["calls"] += 1
                _write(runs_dir, days)
                refused = False
    except OSError:
        return False
    if refused:
        _say(f"no hint: today's {cap} requests are spent ({CAP_VAR})")
        return False
    _say(f"asking Jev, request {day['calls']} of {cap} today")
    return True


def _account(runs_dir: Path, tokens: int = 0, failed: bool = False) -> None:
    """Add what a request cost, or that it failed, to today's row."""
    try:
        with _LOCK:
            days = calls(runs_dir)
            day = _day(days)
            day["tokens"] += tokens
            day["dollars"] = round(day["tokens"] * PRICE_PER_TOKEN, 8)
            day["failed"] += int(failed)
            _write(runs_dir, days)
    except OSError:
        pass


def _say(line: str) -> None:
    """One line in the server's log (the journal, on the box). Never the key."""
    print(f"description hint: {line}", flush=True)


# ------------------------------------------------------------------ the notice


PRIVACY = (
    "<h2>The description hint</h2>\n"
    "<p>After you save a game, the site sends that game's description and its rules to "
    "TypeSafe (typesafe.ai), whose Jev model reads each sentence of the description "
    "against the rules. The design page then shows, as a hint, any sentence that seems "
    "not to match. Only the game is sent &mdash; not your handle, your email address or "
    "your cookie &mdash; and the answer is kept beside the game on the site so the same "
    "description is not sent twice. The hint is optional: the site saves, lists and "
    "plays your game the same way without it.</p>\n"
)


def privacy_paragraph() -> str:
    """What `/privacy` says about the hint: the paragraph while it is on, nothing when off."""
    return PRIVACY if enabled() else ""
