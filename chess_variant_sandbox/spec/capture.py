"""What a capture does, as one ordered list: ``rules.capture`` (R-L4b stage 3).

`docs/reference/capture-effects-design.md` §2 and §4. The list is an ordered list of entries,
each one thing a capture does (``"do"``: ``pay``, ``hand``, ``blast``, ``strike``,
``contacts``), each with its old rule's keys and three new ones: ``on`` (which causes of a
taking it reads), and on a blast ``spread`` and ``spares_taken``. ``path_by`` names the
sweepers whose swept pieces a ``"path"`` reading takes (absent: every sweeper); it is
§2.1's ``by``, renamed because a strike's and a blast's own ``by`` already name whose
captures they act on (A1, schema 10).

**Two spellings, one game.** The old keys (``rules.strike``, ``rules.contacts``,
``rules.explode_on_capture``, ``drops.from_captures``, ``purse.bounty_to`` and
``moves[].effects``) say the list in its default order, ``pay``, ``hand``, ``blast``,
``strike``, ``contacts``, with ``on``, ``spread`` and ``spares_taken`` at what the two
old switches can say. `capture_list` reads either spelling as the list (the one reader);
`capture_as_old_keys` writes a list back as the old keys where they can say it, which the
normal form and the converter both do, so a list and its old spelling are one game under
one name and no stored spec moves. A list only this key can say (another order, an ``on``
the switches cannot say, a kind twice) is its own spelling and its own game.

**Two spellings, kept** (stage 4, `capture-effects-design.md` §6.4). Every surface reads a
capture rule through here: a kind's rule through `rules_as_old_keys` (the list's projection),
and what only the list says (`says_more_than_old_keys`, `list_features`, `list_shape`) from
`capture_list` itself. The old keys stay the stored spelling of every list they can say, so
no file moved; only a list the old keys cannot say is written as ``rules.capture``. The three
fingerprinted metrics sources (`complexity`, `reachability`, `tournament/analysis`) read the
spelling themselves, as they read the portals, with their tables held to these by
`tests/test_capture_surfaces.py`.
"""

from __future__ import annotations

import copy
import json
from typing import Any

#: The causes of a taking, in the order of their bits in the core (`tables.rs`'s ``ON_*``).
CAPTURE_CAUSES: tuple[str, ...] = ("landing", "path", "blast", "push", "collision")

#: The entry kinds, in the default order, which is each kind's code across the boundary.
CAPTURE_KINDS: tuple[str, ...] = ("pay", "hand", "blast", "strike", "contacts")

#: §2.1's default ``on`` per kind: each kind's causes as the old keys read them.
CAPTURE_DEFAULT_ON: dict[str, frozenset[str]] = {
    "pay": frozenset(CAPTURE_CAUSES),
    "hand": frozenset({"landing", "path", "push", "collision"}),
    "blast": frozenset({"landing", "path", "collision"}),
    "strike": frozenset({"landing", "collision"}),
    "contacts": frozenset({"landing"}),
}

#: The keys every entry may carry, and each kind's own: its old rule's keys.
CAPTURE_LIST_KEYS = frozenset({"do", "on", "path_by"})
CAPTURE_BLAST_KEYS = frozenset({"spread", "spares_taken"})
_EXPLODE_INNER = frozenset({"radius", "immune", "by", "of", "blast"})
_STRIKE_INNER = frozenset({"effect", "chain", "position", "immune", "landing", "switches_sides", "by"})
_CONTACTS_INNER = frozenset({"links", "position", "immune", "landing"})
CAPTURE_ENTRY_KEYS: dict[str, frozenset[str]] = {
    "pay": frozenset({"to"}),
    "hand": frozenset(),
    "blast": _EXPLODE_INNER | CAPTURE_BLAST_KEYS,
    "strike": _STRIKE_INNER,
    "contacts": _CONTACTS_INNER,
}
#: The keys of an entry that state its kind's rule (the old key's), as against the list's.
_INNER: dict[str, frozenset[str]] = {
    "pay": frozenset({"to"}), "hand": frozenset(), "blast": _EXPLODE_INNER,
    "strike": _STRIKE_INNER, "contacts": _CONTACTS_INNER,
}
#: The old keys under ``rules`` whose meaning the list carries, written beside it as two
#: spellings of one statement. ``drops.from_captures``, ``purse.bounty_to`` and
#: ``moves[].effects`` are the three nested ones.
CAPTURE_OLD_RULES = ("strike", "contacts", "explode_on_capture")


# ------------------------------------------------------------------------------ readers


def _rules(spec: Any) -> dict[str, Any]:
    rules = spec.get("rules") if isinstance(spec, dict) else None
    return rules if isinstance(rules, dict) else {}


def _pieces(spec: Any) -> list[dict[str, Any]]:
    pieces = spec.get("pieces") if isinstance(spec, dict) else None
    return [p for p in pieces if isinstance(p, dict)] if isinstance(pieces, list) else []


def capture_written(spec: Any) -> bool:
    """Whether ``spec`` writes ``rules.capture`` (the list spelling)."""
    return "capture" in _rules(spec)


def sweeper_ids(spec: Any) -> list[str]:
    """The pieces with a sweep move, in ``pieces`` order: whose swept pieces ``path`` reads."""
    return [
        piece["id"] for piece in _pieces(spec)
        if isinstance(piece.get("id"), str) and isinstance(piece.get("moves"), list)
        and any(isinstance(m, dict) and m.get("kind") == "sweep" for m in piece["moves"])
    ]


def _sweeps_disagree(spec: Any) -> bool:
    for piece in _pieces(spec):
        moves = piece.get("moves")
        sweeps = [m for m in moves if isinstance(m, dict) and m.get("kind") == "sweep"] if isinstance(
            moves, list) else []
        if len({m.get("effects", "landing") for m in sweeps}) > 1:
            return True
    return False


def _path_readers(spec: Any) -> list[str]:
    """The sweepers whose sweeps say ``effects: "path"`` (`schema.sweep_effects`)."""
    from chess_variant_sandbox.spec.schema import sweep_effects

    sweepers = set(sweeper_ids(spec))
    return [
        p["id"] for p in _pieces(spec)
        if isinstance(p.get("id"), str) and p["id"] in sweepers and sweep_effects(p) == "path"
    ]


def _moves_of(piece: dict[str, Any]) -> list[Any]:
    """A piece's moves where they are a list (a malformed spec's are the validator's to name)."""
    moves = piece.get("moves")
    return moves if isinstance(moves, list) else []


def bounty_declared(spec: Any) -> bool:
    """Whether a taking can pay (what a ``pay`` entry pays): a piece type's bounty, or any wild
    piece, whose route pays its own bounty or else its type's value (`compile._compile_npcs`;
    the core's `Tables::pays` is the same test)."""
    def positive(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    npcs = _rules(spec).get("npcs")
    return any(positive(p.get("bounty")) for p in _pieces(spec)) or (
        isinstance(npcs, list) and any(isinstance(n, dict) for n in npcs)
    )


def _explosion_live(explosion: Any) -> bool:
    """A blast the compiler keeps: an object, neither ``by`` nor ``of`` empty."""
    return isinstance(explosion, dict) and bool(explosion) and all(
        explosion.get(key) != [] for key in ("by", "of")
    )


def _hand_fills(rules: dict[str, Any]) -> bool:
    drops = rules.get("drops")
    return isinstance(drops, dict) and bool(drops) and drops.get("from_captures", True) is not False


def _on(kind: str, causes: Any) -> list[str]:
    """``causes`` as a list in `CAPTURE_CAUSES` order (the written order states nothing)."""
    stated = set(causes) if isinstance(causes, list) else set(CAPTURE_DEFAULT_ON[kind])
    return [cause for cause in CAPTURE_CAUSES if cause in stated]


def _from_old_keys(spec: Any) -> list[dict[str, Any]] | None:
    """The list the old keys say, in the default order; ``None`` where they say something no
    list says: a piece whose sweeps disagree on ``effects`` (W1.disagree), or a blast spread
    on the path of some sweepers and not others beside a strike or contacts that reads no
    path of them (one ``effects`` switch per piece, two readings)."""
    from chess_variant_sandbox.spec.schema import blast_strikes, strike_is_live

    rules = _rules(spec)
    if _sweeps_disagree(spec):
        return None
    sweepers = sweeper_ids(spec)
    readers = _path_readers(spec)
    everyone = readers == sweepers
    strikes = blast_strikes(rules)
    entries: list[dict[str, Any]] = []
    if bounty_declared(spec):
        pay: dict[str, Any] = {"do": "pay"}
        purse = rules.get("purse")
        if isinstance(purse, dict) and purse.get("bounty_to") == "loser":
            pay["to"] = "loser"
        entries.append(pay)
    if _hand_fills(rules):
        entries.append({"do": "hand"})
    explosion = rules.get("explode_on_capture")
    if _explosion_live(explosion):
        assert isinstance(explosion, dict)
        if readers and not everyone:
            return None
        blast = {"do": "blast", **{k: v for k, v in explosion.items() if k != "strikes"}}
        if readers:
            blast["spread"] = True
        if strikes in ("target", "both"):
            blast["spares_taken"] = True
        entries.append(blast)
    for kind in ("strike", "contacts"):
        rule = rules.get(kind)
        live = strike_is_live(rule) if kind == "strike" else (
            isinstance(rule, dict) and bool(rule.get("links"))
        )
        if not live:
            continue
        assert isinstance(rule, dict)
        on = set(CAPTURE_DEFAULT_ON[kind])
        if strikes in ("blast", "both") and _explosion_live(explosion):
            on.add("blast")
        entry: dict[str, Any] = {"do": kind, **rule}
        if readers:
            on.add("path")
            if not everyone:
                entry["path_by"] = list(readers)
        entry["on"] = _on(kind, sorted(on))
        entries.append(entry)
    for entry in entries:
        entry.setdefault("on", _on(entry["do"], None))
    return entries


def _as_written(spec: Any) -> list[dict[str, Any]] | None:
    """``rules.capture`` with each entry in one shape: ``on`` always stated, in cause order;
    ``path_by`` in ``pieces`` order and left out where it names every sweeper; a blast's
    ``spread`` and ``spares_taken`` left out where false. ``None`` if malformed."""
    written = _rules(spec).get("capture")
    if not isinstance(written, list):
        return None
    sweepers = sweeper_ids(spec)
    order = {pid: index for index, pid in enumerate(p.get("id") for p in _pieces(spec) if isinstance(p.get("id"), str))}
    out: list[dict[str, Any]] = []
    for entry in written:
        if not isinstance(entry, dict) or entry.get("do") not in CAPTURE_KINDS:
            return None
        kind = entry["do"]
        shaped = {k: copy.deepcopy(v) for k, v in entry.items() if k not in ("on", "path_by")}
        if entry.get("on") is not None and not isinstance(entry.get("on"), list):
            return None
        shaped["on"] = _on(kind, entry.get("on"))
        by = entry.get("path_by")
        if isinstance(by, list) and all(isinstance(pid, str) for pid in by):
            by = sorted(set(by), key=lambda pid: order.get(pid, len(order)))
            if by != sweepers:
                shaped["path_by"] = by
        elif by is not None:
            return None
        for key in CAPTURE_BLAST_KEYS:
            if shaped.get(key) is False:
                del shaped[key]
        if kind == "pay" and shaped.get("to") == "taker":
            del shaped["to"]  # `purse.bounty_to`'s off value
        out.append(shaped)
    return out


def capture_list(spec: Any) -> list[dict[str, Any]] | None:
    """What a capture does in ``spec``, as the list: **the one reader** of both spellings.

    ``rules.capture`` as written, each entry in one shape (`_as_written`), or the old keys'
    list in the default order (`_from_old_keys`). Empty is plain capture. ``None`` where the
    spelling is malformed, or the old keys say something no list says."""
    if capture_written(spec):
        return _as_written(spec)
    return _from_old_keys(spec)


def _same(one: Any, other: Any) -> bool:
    return json.dumps(one, sort_keys=True) == json.dumps(other, sort_keys=True)


# ------------------------------------------------------------------------- the old keys


def _without_old_keys(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with ``rules.capture`` and every old capture key taken out."""
    out = copy.deepcopy(spec)
    rules = out.get("rules")
    if not isinstance(rules, dict):
        return out
    rules.pop("capture", None)
    for key in CAPTURE_OLD_RULES:
        rules.pop(key, None)
    if isinstance(rules.get("drops"), dict):
        rules["drops"].pop("from_captures", None)
    if isinstance(rules.get("purse"), dict):
        rules["purse"].pop("bounty_to", None)
    for piece in _pieces(out):
        for move in _moves_of(piece):
            if isinstance(move, dict):
                move.pop("effects", None)
    return out


def _write_rules(out: dict[str, Any], entries: list[dict[str, Any]]) -> bool:
    """Each kind's rule from its first entry, written as the old keys into ``out``; False if a
    kind's rule has nowhere to go (a hand without ``rules.drops``, a pay to the loser with
    no purse)."""
    rules = out.setdefault("rules", {})
    seen: set[str] = set()
    for entry in entries:
        kind = entry["do"]
        if kind in seen:
            continue
        seen.add(kind)
        inner = {k: copy.deepcopy(v) for k, v in entry.items() if k in _INNER[kind]}
        if kind == "pay":
            if inner.get("to", "taker") != "taker":
                if not isinstance(rules.get("purse"), dict):
                    return False
                rules["purse"]["bounty_to"] = inner["to"]
        elif kind == "hand":
            if not isinstance(rules.get("drops"), dict):
                return False
            # Written out, as every stored hand game writes it: absent means true too, but
            # the normal form keeps the two spellings' names apart.
            rules["drops"]["from_captures"] = True
        elif kind == "blast":
            rules["explode_on_capture"] = inner
        else:
            rules[kind] = inner
    if "hand" not in seen and isinstance(rules.get("drops"), dict) and rules["drops"]:
        rules["drops"]["from_captures"] = False
    return True


def capture_as_old_keys(spec: Any) -> dict[str, Any] | None:
    """``spec`` with ``rules.capture`` written back as the old keys, where they say the same
    list; else ``None`` (and ``None`` for a spec that writes no list).

    The old keys say a list when it holds each kind at most once, in the default order, with
    ``on``, ``spread`` and ``spares_taken`` at what ``explode_on_capture.strikes`` and
    ``moves[].effects`` can say. Asked by the one test that settles it: the old keys written
    out read back (`_from_old_keys`) as the very list written."""
    if not isinstance(spec, dict) or not capture_written(spec):
        return None
    written = _as_written(spec)
    if written is None:
        return None
    kinds = [entry["do"] for entry in written]
    if len(set(kinds)) != len(kinds) or kinds != sorted(kinds, key=CAPTURE_KINDS.index):
        return None
    out = _without_old_keys(spec)
    if not _write_rules(out, written):
        return None
    rules = out["rules"]
    by_kind = {entry["do"]: entry for entry in written}
    blast = by_kind.get("blast")
    meets = [by_kind[k] for k in ("strike", "contacts") if k in by_kind]
    if blast is not None:
        spares = bool(blast.get("spares_taken"))
        struck = any("blast" in entry["on"] for entry in meets)
        strikes = {(True, True): "both", (True, False): "target", (False, True): "blast"}.get(
            (spares, struck))
        if strikes is not None:
            rules["explode_on_capture"]["strikes"] = strikes
    readers: list[str] = []
    for entry in meets:
        if "path" in entry["on"]:
            readers = entry.get("path_by", sweeper_ids(out))
            break
    else:
        if blast is not None and blast.get("spread"):
            readers = sweeper_ids(out)
    for piece in _pieces(out):
        if piece.get("id") in readers:
            for move in _moves_of(piece):
                if isinstance(move, dict) and move.get("kind") == "sweep":
                    move["effects"] = "path"
    if not rules:
        out.pop("rules")
    back = _from_old_keys(out)
    return out if back is not None and _same(back, written) else None


def as_list_spelling(spec: dict[str, Any]) -> dict[str, Any] | None:
    """``spec`` with its old capture keys written as ``rules.capture`` (each entry's ``on`` at
    its default left out), or ``None`` where the old keys say no list or no capture rule at
    all. The spelling a person may write; the converter and the normal form write it back."""
    entries = capture_list(spec)
    if not entries or capture_written(spec):
        return None
    out = _without_old_keys(spec)
    for entry in entries:
        if set(entry["on"]) == CAPTURE_DEFAULT_ON[entry["do"]]:
            del entry["on"]
    out.setdefault("rules", {})["capture"] = entries
    return out


def normal_entries(spec: dict[str, Any]) -> list[dict[str, Any]] | None:
    """A list the old keys cannot say, in its hashing shape: in written order (the order is the
    meaning), each entry as `_as_written` shapes it with an ``on`` at its kind's default left
    out, and a contact link's ``own: false`` (its off value) left out."""
    written = _as_written(spec)
    if written is None:
        return None
    for entry in written:
        if set(entry["on"]) == CAPTURE_DEFAULT_ON[entry["do"]]:
            del entry["on"]
        for link in entry.get("links") or [] if entry["do"] == "contacts" else []:
            if isinstance(link, dict) and link.get("own") is False:
                del link["own"]
    return written


def capture_base(spec: dict[str, Any]) -> dict[str, Any]:
    """The rules a written list walks, as the old keys: each kind's rule from its first entry,
    and none of the two switches (the list says what they said). What the validator checks a
    list's rules against, and what the compiler builds the tables from; the list then crosses
    beside them (`compile.with_capture_list`)."""
    out = _without_old_keys(spec)
    written = _as_written(spec) or []
    _write_rules(out, written)
    if isinstance(out.get("rules"), dict) and not out["rules"]:
        out.pop("rules")
    return out


def compile_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The list as the compiler takes it (`compile.with_capture_list`): ``do``, ``on``,
    ``spread``, ``spares_taken`` and ``by`` (the sweepers a path reading takes)."""
    out: list[dict[str, Any]] = []
    for entry in entries:
        item: dict[str, Any] = {"do": entry["do"], "on": list(entry["on"])}
        for key in CAPTURE_BLAST_KEYS:
            if entry.get(key):
                item[key] = True
        if "path_by" in entry:
            item["by"] = list(entry["path_by"])
        out.append(item)
    return out


# ---------------------------------------------------------------- the pairs the list lifts


def lifted_pairs(spec: Any) -> list[str]:
    """The pairs of §3.1 and §3.2 (`capture-effects-design.md`) that ``spec``'s old keys state,
    by their `refusals.toml` id: each was refused until R-L4b, and each plays only as the
    list (the core's default-order walk plays every other game, stage 1). The compiler
    hands such a spec's list to the core (`compile.compile_spec`)."""
    from chess_variant_sandbox.spec.schema import (
        blast_strikes,
        contacts_only_remove,
        move_push,
        purse_upkeep,
        strike_is_live,
        strike_only_removes,
    )

    rules = _rules(spec)
    strike = rules.get("strike") if strike_is_live(rules.get("strike")) else None
    contacts = rules.get("contacts")
    if not (isinstance(contacts, dict) and contacts.get("links")):
        contacts = None
    blast = _explosion_live(rules.get("explode_on_capture"))
    strikes = blast_strikes(rules) if blast else "none"
    drops = rules.get("drops") if isinstance(rules.get("drops"), dict) else None
    hand = bool(drops)
    either_face = bool(drops and drops.get("either_face") is True)
    gating = bool(drops and "gating" in drops)
    moves = [m for p in _pieces(spec) for m in _moves_of(p) if isinstance(m, dict)]
    sweep = any(m.get("kind") == "sweep" for m in moves)
    push = any(move_push(m) is not None for m in moves)
    upkeep = purse_upkeep(rules) is not None
    switches = isinstance(strike, dict) and bool(strike.get("switches_sides"))
    found = {
        "T1.position": strike is not None and strikes != "none"
        and strike.get("position", "advance") not in ("stay", "home"),
        "T1.choose": strike is not None and blast and strike.get("position") == "choose",
        "T4.choose": contacts is not None and blast and contacts.get("position") == "choose",
        "T2": strike is not None and hand and not strike_only_removes(strike),
        "T5": contacts is not None and hand and not contacts_only_remove(contacts),
        "T3": strike is not None and contacts is not None,
        "T4.switch": contacts is not None and strikes != "none",
        "H4": strike is not None and either_face and strike.get("position", "advance") != "advance",
        "H5": either_face and switches,
        "W3.both": sweep and blast and (strike is not None or contacts is not None),
        "push.X2.strike": push and strike is not None,
        "push.X2.contacts": push and contacts is not None,
        "push.X3": push and blast,
        "gating.blast": gating and blast,
        "gating.captures": gating and _hand_fills(rules),
        "upkeep.U9": upkeep and contacts is not None,
        "upkeep.U10": upkeep and switches,
    }
    return [rid for rid, present in found.items() if present]


# ------------------------------------------------------------------------- the validator

def _in_rounds(spec: dict[str, Any]) -> list[str]:
    """What a game in rounds cannot walk (`refusals.py`'s T1 row for ``rules.capture``):
    R-L4b5 walks the list over a round's takings (`rounds.md` §4 steps 5 and 6), but a strike
    or link placed ``stay`` is written before the list (a shot at step 0, a contact in place
    at step 4), and a round's own merges and shots read one table, so neither beside a list
    the core walks. ``spec`` is the rules as the old keys (the spec, or `capture_base`)."""
    from chess_variant_sandbox.spec.refusals import refusal
    from chess_variant_sandbox.spec.schema import is_simultaneous, strike_is_live

    rules = _rules(spec)
    if not is_simultaneous(rules):
        return []
    strike = rules.get("strike") if strike_is_live(rules.get("strike")) else None
    contacts = rules.get("contacts")
    if not (isinstance(contacts, dict) and contacts.get("links")):
        contacts = None
    stays = any(isinstance(rule, dict) and rule.get("position") == "stay" for rule in (strike, contacts))
    if stays or (strike is not None and contacts is not None):
        return [refusal("T1", "rules.capture")]
    return []


def capture_walk_errors(spec: dict[str, Any], errors: list[str]) -> list[str]:
    """Old keys stating a pair only the capture list plays (`lifted_pairs`): they must say a
    list (`capture_list`), and in a round no strike or link that stays and no second table
    (`_in_rounds`). ``errors`` is what the validator found so far."""
    pairs = lifted_pairs(spec)
    if not pairs:
        return []
    found: list[str] = []
    if capture_list(spec) is None and not any("sweeps disagree" in e for e in errors):
        found.append(
            f"{', '.join(pairs)} beside a blast spread on the path of some sweepers and not "
            "others: one effects switch per piece cannot say which entry reads whose path; "
            "write it as rules.capture"
        )
    found.extend(_in_rounds(spec))
    return found


def capture_list_errors(spec: dict[str, Any]) -> list[str]:
    """A spec that writes ``rules.capture``: the list's own shape first; then the rules its
    entries state, as the old keys: the very old keys where they say the list
    (`capture_as_old_keys`), else each kind's rule without the two switches
    (`capture_base`), which the list replaces."""
    from chess_variant_sandbox.spec.schema import (
        BOUNTY_PAYEES,
        READING_A_LIST,
        _check_keys,
        validate_spec,
    )

    errors: list[str] = []
    rules = _rules(spec)
    written = rules["capture"]
    if not isinstance(written, list):
        return ['rules.capture must be a list of entries, each an object with a "do"']
    for key in CAPTURE_OLD_RULES:
        if key in rules:
            errors.append(
                f"rules.{key} beside rules.capture: two spellings of what a capture does; "
                "write it as an entry of the list"
            )
    drops, purse = rules.get("drops"), rules.get("purse")
    if isinstance(drops, dict) and "from_captures" in drops:
        errors.append(
            'rules.drops.from_captures beside rules.capture: a hand entry ({"do": "hand"}) '
            "says it; leave from_captures out"
        )
    if isinstance(purse, dict) and "bounty_to" in purse:
        errors.append(
            "rules.purse.bounty_to beside rules.capture: a pay entry's to says it; leave "
            "bounty_to out"
        )
    pieces = _pieces(spec)
    if any(isinstance(m, dict) and "effects" in m for p in pieces for m in _moves_of(p)):
        errors.append(
            "pieces[].moves[].effects beside rules.capture: an entry's on ('path') and path_by "
            "say which sweeps it reads; leave effects out"
        )
    ids = {p["id"] for p in pieces if isinstance(p.get("id"), str)}
    sweepers = set(sweeper_ids(spec))
    seen: set[str] = set()
    for index, entry in enumerate(written):
        where = f"rules.capture[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object")
            continue
        kind = entry.get("do")
        if kind not in CAPTURE_KINDS:
            errors.append(f"{where}.do must be one of {CAPTURE_KINDS}, got {kind!r}")
            continue
        _check_keys(entry, set(CAPTURE_LIST_KEYS | CAPTURE_ENTRY_KEYS[kind]), where, errors)
        on = entry.get("on")
        if on is not None:
            if not isinstance(on, list) or not all(c in CAPTURE_CAUSES for c in on):
                errors.append(f"{where}.on must be a list of {CAPTURE_CAUSES}, got {on!r}")
                on = None
            elif not on:
                errors.append(f"{where}.on is empty: the entry reads no taking; leave it out")
        causes = set(on) if isinstance(on, list) else set(CAPTURE_DEFAULT_ON[kind])
        by = entry.get("path_by")
        if by is not None:
            if not isinstance(by, list) or not by or not all(isinstance(x, str) for x in by):
                errors.append(f"{where}.path_by must be a non-empty list of piece ids")
            else:
                for pid in by:
                    if pid not in ids:
                        errors.append(f"{where}.path_by names {pid!r}, which is not a piece")
                    elif pid not in sweepers:
                        errors.append(
                            f"{where}.path_by names {pid!r}, which never sweeps, so its path "
                            "is never read"
                        )
                if "path" not in causes:
                    errors.append(
                        f"{where}.path_by without 'path' in its on: it names whose path the "
                        "entry reads, and the entry reads none"
                    )
        if "path" in causes and on is not None and not sweepers:
            errors.append(f"{where}.on reads 'path', and no piece sweeps")
        for key in sorted(CAPTURE_BLAST_KEYS):
            if key in entry and not isinstance(entry[key], bool):
                errors.append(f"{where}.{key} must be true or false")
        if entry.get("spread") is True and not causes & {"path", "push"}:
            errors.append(
                f"{where}.spread centres a blast on each swept or pushed-off piece, and the "
                "entry reads neither 'path' nor 'push'; leave it out"
            )
        if kind == "pay" and entry.get("to", "taker") not in BOUNTY_PAYEES:
            errors.append(f"{where}.to must be one of {BOUNTY_PAYEES}, got {entry.get('to')!r}")
        if kind in seen and set(entry) & _INNER[kind]:
            errors.append(
                f"{where} states its kind's rule again: entries of one kind share the first "
                f"{kind!r} entry's rule, so write its keys there and only on and path_by "
                "(and a blast's spread and spares_taken) here"
            )
        seen.add(kind)
        if kind == "hand" and not isinstance(drops, dict):
            errors.append(f"{where}: a hand entry files taken pieces in rules.drops; give the hand")
        if kind == "pay" and entry.get("to") == "loser" and not isinstance(purse, dict):
            errors.append(f"{where}.to 'loser' pays into a purse; give rules.purse")
    if bounty_declared(spec) and "pay" not in seen:
        errors.append(
            "a bounty (pieces[].bounty, or a wild piece's) beside rules.capture with no pay "
            'entry is never paid; add {"do": "pay"}'
        )
    if "pay" in seen and not bounty_declared(spec):
        errors.append("a pay entry in rules.capture with no bounty declared pays nothing; leave it out")
    if errors:
        return errors
    lowered = capture_as_old_keys(spec)
    if lowered is not None:
        return validate_spec(lowered)
    errors.extend(_in_rounds(capture_base(spec)))
    token = READING_A_LIST.set(True)
    try:
        errors.extend(validate_spec(capture_base(spec)))
    finally:
        READING_A_LIST.reset(token)
    return list(dict.fromkeys(errors))


# ------------------------------------------------------------------------- the surfaces


def says_more_than_old_keys(spec: Any) -> bool:
    """Whether ``spec`` writes a list the old keys cannot say: another order, an ``on`` the
    two switches cannot say, or a kind twice (`capture_as_old_keys` is ``None``). Only such
    a spec is stored as ``rules.capture``; every other list is written back as the old keys."""
    return isinstance(spec, dict) and capture_written(spec) and capture_as_old_keys(spec) is None


def rules_as_old_keys(spec: Any) -> Any:
    """``spec`` as a surface that reads a kind's rule sees it, **from the one list**.

    A spec in the old spelling is returned as it is (the old keys *are* its list in the
    default order); a written list the old keys say is returned lowered
    (`capture_as_old_keys`); a list only ``rules.capture`` says is returned as each kind's
    rule from its first entry (`capture_base`), the two switches left out, since the list
    says what they said. What that projection cannot carry (the order, each entry's ``on``,
    ``path_by``, ``spread``, ``spares_taken``, a kind twice) a surface reads from
    `capture_list` itself; `says_more_than_old_keys` says when there is any. Every surface
    that states, prices, compares, edits or translates a capture rule reads it through here
    (the rules text, plain words, the relations, the distance, the operators, the prompt's
    reading, Fairy-Stockfish, the page), so no surface reads the old keys of a list."""
    if not isinstance(spec, dict) or not capture_written(spec):
        return spec
    lowered = capture_as_old_keys(spec)
    if lowered is not None:
        return lowered
    if _as_written(spec) is None:
        return spec
    return capture_base(spec)


def list_shape(spec: Any) -> list[dict[str, Any]]:
    """What `rules_as_old_keys` cannot carry, for a list only ``rules.capture`` says: each
    entry's kind and the list's own keys (``on`` at its default left out, ``path_by``,
    ``spread``, ``spares_taken``), in written order; empty for every other spec. The relations
    and the distance compare it beside each kind's rule."""
    if not says_more_than_old_keys(spec):
        return []
    entries = normal_entries(spec) or []
    keep = CAPTURE_LIST_KEYS | CAPTURE_BLAST_KEYS
    return [{k: v for k, v in entry.items() if k in keep} for entry in entries]


def capture_rule(spec: Any, kind: str) -> dict[str, Any] | None:
    """The rule of ``kind`` in ``spec``'s list (the first entry of that kind, its own keys
    only, as its old key wrote it), or ``None`` where the list has no such entry. A blast's
    ``strikes`` switch is not among its keys: the list says it as ``spares_taken`` and the
    strike's or contacts' ``on``."""
    for entry in capture_list(spec) or []:
        if entry["do"] == kind:
            return {k: copy.deepcopy(v) for k, v in entry.items() if k in _INNER[kind]}
    return None


def capture_kinds(spec: Any) -> list[str]:
    """The kinds ``spec``'s list holds, each once, in the order their first entries come."""
    return list(dict.fromkeys(entry["do"] for entry in capture_list(spec) or []))


def _sayable_on(kind: str) -> list[frozenset[str]]:
    """The on sets the old keys can say for ``kind``: `pay`, `hand` and `blast` only their
    default; a strike or contacts its default with ``"path"`` (a sweep's ``effects``) or
    ``"blast"`` (``explode_on_capture.strikes``) added."""
    base = CAPTURE_DEFAULT_ON[kind]
    if kind not in ("strike", "contacts"):
        return [base]
    return [base, base | {"path"}, base | {"blast"}, base | {"path", "blast"}]


def list_features(spec: Any) -> set[str]:
    """What ``spec``'s written list says that the old keys cannot, by name: ``"order"`` (the
    kinds not in the default order), ``"twice"`` (a kind in two entries), ``"on"`` (an entry
    reading causes the two switches cannot say). Read as the game plays it: a reading of a
    cause the game never makes is no reading (`as_played_old_keys`). Empty for a spec in the
    old spelling and for a list the old keys say. The feature inventory and the operators
    read it."""
    if not says_more_than_old_keys(spec) or as_played_old_keys(spec) is not None:
        return set()
    written = _as_written(spec) or []
    possible = possible_causes(spec)
    kinds = [entry["do"] for entry in written]
    firsts = list(dict.fromkeys(kinds))
    found: set[str] = set()
    if firsts != sorted(firsts, key=CAPTURE_KINDS.index):
        found.add("order")
    if len(firsts) != len(kinds):
        found.add("twice")
    if any(
        frozenset(entry["on"]) & possible not in {on & possible for on in _sayable_on(entry["do"])}
        for entry in written
    ):
        found.add("on")
    return found


def possible_causes(spec: Any) -> frozenset[str]:
    """The causes of a taking ``spec`` can make: a landing always, a sweep's path where a piece
    sweeps, a push where a move pushes, a collision in a game in rounds, a blast where the list
    has a blast. A cause outside these is never read, whatever an entry's ``on`` says."""
    from chess_variant_sandbox.spec.schema import is_simultaneous, move_push

    moves = [m for p in _pieces(spec) for m in _moves_of(p) if isinstance(m, dict)]
    found = {"landing"}
    if sweeper_ids(spec):
        found.add("path")
    if any(move_push(m) is not None for m in moves):
        found.add("push")
    if is_simultaneous(_rules(spec)):
        found.add("collision")
    if any(entry["do"] == "blast" for entry in capture_list(spec) or []):
        found.add("blast")
    return frozenset(found)


def as_played_old_keys(spec: Any) -> dict[str, Any] | None:
    """``spec``'s written list as the old keys **as this game plays it**: each entry's ``on``
    read only over the causes the game can make (`possible_causes`), the rest at its kind's
    default, then lowered (`capture_as_old_keys`). A list that differs from the old keys only
    in a cause the game never makes is the same play, so a translator may re-express it as
    them; ``None`` where the list still says what they cannot (an order, a kind twice, a
    reading of a cause the game makes), or writes no list."""
    if not isinstance(spec, dict) or not capture_written(spec):
        return None
    written = _as_written(spec)
    if written is None:
        return None
    possible = possible_causes(spec)
    out = copy.deepcopy(spec)
    entries = []
    for entry in written:
        default = CAPTURE_DEFAULT_ON[entry["do"]]
        entry = dict(entry)
        entry["on"] = _on(entry["do"], sorted((set(entry["on"]) & possible) | (default - possible)))
        entries.append(entry)
    out["rules"]["capture"] = entries
    return capture_as_old_keys(out)


def default_order(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with its written list put in the default order, each kind's entries merged
    into one reading every cause any of them read (the first entry's rule kept), and written
    back as the old keys where they then say it. The operators' way back from a list only
    ``rules.capture`` says; a spec in the old spelling is returned unchanged."""
    if not capture_written(spec):
        return spec
    written = _as_written(spec)
    if written is None:
        return spec
    merged: dict[str, dict[str, Any]] = {}
    for entry in written:
        kind = entry["do"]
        if kind not in merged:
            merged[kind] = copy.deepcopy(entry)
            continue
        first = merged[kind]
        first["on"] = _on(kind, sorted(set(first["on"]) | set(entry["on"])))
        for key in CAPTURE_BLAST_KEYS:
            if entry.get(key):
                first[key] = True
    entries = [merged[kind] for kind in CAPTURE_KINDS if kind in merged]
    out = copy.deepcopy(spec)
    out["rules"]["capture"] = entries
    lowered = capture_as_old_keys(out)
    return lowered if lowered is not None else out


def with_list(spec: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """``spec`` with its capture rules replaced by ``entries`` (each with its kind's rule on
    the first entry of a kind), written back as the old keys where they say it, else as
    ``rules.capture`` with each ``on`` at its default left out. No entries is plain capture."""
    out = _without_old_keys(spec)
    if not entries:
        if isinstance(out.get("rules"), dict):
            drops = out["rules"].get("drops")
            if isinstance(drops, dict) and drops:
                drops["from_captures"] = False
        return out
    written = []
    rule_of: dict[str, dict[str, Any]] = {}
    for entry in entries:  # each kind's rule, wherever its entries stand
        rule_of.setdefault(entry["do"], {})
        for key, value in entry.items():
            if key in _INNER[entry["do"]]:
                rule_of[entry["do"]].setdefault(key, copy.deepcopy(value))
    seen: set[str] = set()
    for entry in entries:
        entry = {k: copy.deepcopy(v) for k, v in entry.items() if k not in _INNER[entry["do"]]}
        if entry["do"] not in seen:  # the rule rides on the kind's first entry
            seen.add(entry["do"])
            entry.update(rule_of[entry["do"]])
        if set(entry.get("on", CAPTURE_DEFAULT_ON[entry["do"]])) == CAPTURE_DEFAULT_ON[entry["do"]]:
            entry.pop("on", None)
        written.append(entry)
    out.setdefault("rules", {})["capture"] = written
    lowered = capture_as_old_keys(out)
    return lowered if lowered is not None else out


def carry_list(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """An edit made to the projection of ``parent`` (`rules_as_old_keys`), written back as
    ``parent``'s list: each kind's rule as the edit left it, in ``parent``'s order with its
    entries' ``on``, ``path_by``, ``spread`` and ``spares_taken``; a kind the edit removed
    goes, one it added comes in at its default place. A cause the child can no longer make,
    and a sweeper it no longer has, are dropped from the readings, and an entry left reading
    nothing goes. ``child`` unchanged where ``parent`` writes no list only the key says."""
    if not says_more_than_old_keys(parent):
        return child
    edited = capture_list(child)
    written = _as_written(parent)
    if edited is None or written is None:
        return child
    rules = {entry["do"]: {k: v for k, v in entry.items() if k in _INNER[entry["do"]]} for entry in edited}
    probe = with_list(child, [{"do": kind, **rule} for kind, rule in rules.items()])
    possible = possible_causes(probe)
    sweepers = set(sweeper_ids(child))
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in written:
        kind = entry["do"]
        if kind not in rules:
            continue
        on = [cause for cause in entry["on"] if cause in possible]
        if not on:
            continue
        item: dict[str, Any] = {"do": kind}
        if kind not in seen:
            item.update(copy.deepcopy(rules[kind]))
            seen.add(kind)
        item["on"] = _on(kind, on)
        by = [pid for pid in entry.get("path_by", []) if pid in sweepers]
        if by and "path" in on:
            item["path_by"] = by
        for key in CAPTURE_BLAST_KEYS:
            if entry.get(key) and (key != "spread" or set(on) & {"path", "push"}):
                item[key] = True
        entries.append(item)
    for kind in CAPTURE_KINDS:
        if kind in rules and kind not in seen:
            at = next(
                (i for i, e in enumerate(entries) if CAPTURE_KINDS.index(e["do"]) > CAPTURE_KINDS.index(kind)),
                len(entries),
            )
            entries.insert(at, {"do": kind, **copy.deepcopy(rules[kind])})
    return with_list(child, entries)


def needs_the_walk(spec: Any) -> bool:
    """Whether the core must be handed ``spec``'s list (`capture_list`) rather than derive the
    default-order list from the old tables: a written list the old keys cannot say, or old
    keys stating a pair the list lifted (`lifted_pairs`)."""
    if capture_written(spec):
        lowered = capture_as_old_keys(spec)
        return lowered is None or bool(lifted_pairs(lowered))
    return bool(lifted_pairs(spec))
