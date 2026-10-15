"""Every reader of a field an inert-rule refusal stands on, enumerated.

`spec/schema.py::_check_inert_rules` refuses twelve spellings that declare a rule and change
no game. Each was proven by the source standard of
`docs/reference/bugs-investigations.md`, "Rules that declare nothing" (Q56, re-read by
Q111/Q112): name the boundary fields the declaration moves, read every site in `rust/src/`
that reads them, and call the spelling inert only where each reader is the rule its name
describes and a provable no-op under the refused spelling.

**Inert is a property of a (spelling, core) pair, not of a rule.** A reader added to one of
those fields later un-proves a shipped refusal and no game, perft or test notices -- which
is the defect `docs/bugs.md` carried as *Inert-rule refusals are proven against one core*.
So this file does to those fields what
`tests/test_dsl_generality.py::TestNothingInTheCoreWorksOutASquare` does to the board's
width: it reads the source and fails on any read site that is not on `READERS`, each entry
of which says why that site is still a no-op under the refusals that lean on it. Adding a
site is re-running step 3 of the standard for every clause the field serves, and writing
the answer down beside it.

What is walked, and what the walk does not see:

* **The boundary.** Each refused key of `bridge.build_tables`' dict is read into `Tables` by
  `keys::X` in `Tables::from_py` and nowhere else; `TestTheBoundaryReadsThemOnce` holds it.
* **The core.** Every `.member` of `Tables` that carries one of those keys, in every file of
  `rust/src/`, by function. A `Tables` method that reads one is an *accessor*, and the
  callers of it are followed to a fixed point, since "reached through a method rather than
  by name" is the hole the reference names. A struct literal that copies a field into a
  field of another struct (`has_drops: t.drops.is_some()`) is a copy, and the copy's readers
  in that file are followed too.
* **The bridge.** Which `CompiledSpec` attributes `bridge.py` reads for these rules, and
  where its dict keys are spelled, by top-level definition.
* **Not seen:** a copy through a local variable into something unnamed (the site that makes
  it is listed and its reason says what it derives), a destructuring `let Tables { .. }`,
  and anything outside `rust/src/` and `bridge.py` -- `complexity`, `describe`, `fairy/` and
  the mechanics hash read the spec keys too, and those are numbers rather than play (the
  reference's *What the source question does not cover*).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "rust" / "src"
TABLES = CORE / "tables.rs"
BRIDGE = ROOT / "chess_variant_sandbox" / "engine" / "bridge.py"

# The twelve refusals, by the reference's case number, and the boundary keys each one's
# proof reads (its Q112 "Fields" list, plus `promo_on_leaving`, which schema 15 added to the
# promotion half after the list was taken). Case 2 is declined, not refused.
CLAUSES: dict[int, tuple[str, frozenset[str]]] = {
    1: ("en_passant where the promoting piece cannot double-step", frozenset({"ep_type"})),
    3: ("must_capture where nothing can be taken", frozenset({"must_capture"})),
    4: ("explode_on_capture where nothing can be taken", frozenset({
        "has_explosion", "explosion_immune", "explosion_blast", "explosion_blast_index",
        "explosion_by", "explosion_of",
    })),
    5: ("drops where nothing can be taken", frozenset({
        "has_drops", "drops_from_captures", "drops_regions", "drops_no_doubled",
        "drops_no_mate", "drops_stock", "drops_unlimited",
    })),
    6: ("drops with no from_captures and no stock", frozenset({
        "has_drops", "drops_from_captures", "drops_regions", "drops_no_doubled",
        "drops_no_mate", "drops_stock", "drops_unlimited",
    })),
    7: ("bare_royal with no royal piece", frozenset({"has_bare_royal", "bare_royal"})),
    8: ("check_limit with no royal piece", frozenset({"check_limit"})),
    9: ("forbid_check with no royal piece", frozenset({"forbid_check"})),
    10: ('royal_facing "forbidden" with no royal piece', frozenset({
        "royal_facing_forbidden", "facing_index", "facing_entries", "facing_paths",
    })),
    11: ("check_limit beside forbid_check", frozenset({"check_limit", "forbid_check"})),
    12: ("castling with no unmoved rook on a king's rank", frozenset({
        "castle_king", "castle_rook", "castle_steps", "castle_king_files",
        "castle_rook_files",
    })),
    13: ("reach or promotion naming a piece that can never stand", frozenset({
        "has_reach", "reach_piece_type", "reach_squares", "reach_result", "reach_safe",
        "goal_distance", "goal_diameter", "promo_squares", "promo_targets_flat",
        "promo_targets_index", "promo_optional", "promo_piece_types", "promo_on_leaving",
        "ep_type", "entry_promotion", "multi_royal",
    })),
}

REFUSED_KEYS = frozenset().union(*(keys for _, keys in CLAUSES.values()))

# What `Tables::from_py` turns each refused key into. A key read into a local and folded
# into a member (`facing_index` and `facing_entries` into `facing_lines`) maps to the member.
MEMBERS: dict[str, frozenset[str]] = {
    "ep_type": frozenset({"ep_type"}),
    "must_capture": frozenset({"must_capture"}),
    "has_explosion": frozenset({"explosion"}),
    "explosion_immune": frozenset({"explosion"}),
    "explosion_blast": frozenset({"explosion"}),
    "explosion_blast_index": frozenset({"explosion"}),
    "explosion_by": frozenset({"explosion"}),
    "explosion_of": frozenset({"explosion"}),
    "has_drops": frozenset({"drops"}),
    "drops_from_captures": frozenset({"drops"}),
    "drops_regions": frozenset({"drops"}),
    "drops_no_doubled": frozenset({"drops"}),
    "drops_no_mate": frozenset({"drops"}),
    "drops_stock": frozenset({"drops"}),
    "drops_unlimited": frozenset({"drops"}),
    "has_bare_royal": frozenset({"bare_royal"}),
    "bare_royal": frozenset({"bare_royal"}),
    "check_limit": frozenset({"check_limit"}),
    "forbid_check": frozenset({"forbid_check"}),
    "royal_facing_forbidden": frozenset({"royal_facing_forbidden"}),
    "facing_index": frozenset({"facing_lines"}),
    "facing_entries": frozenset({"facing_lines"}),
    "facing_paths": frozenset({"facing_paths"}),
    "castle_king": frozenset({"castling"}),
    "castle_rook": frozenset({"castling"}),
    "castle_steps": frozenset({"castling"}),
    "castle_king_files": frozenset({"castle_files"}),
    "castle_rook_files": frozenset({"castle_files"}),
    "has_reach": frozenset({"reach"}),
    "reach_piece_type": frozenset({"reach"}),
    "reach_squares": frozenset({"reach"}),
    "reach_result": frozenset({"reach"}),
    "reach_safe": frozenset({"reach"}),
    "goal_distance": frozenset({"goal_distance"}),
    "goal_diameter": frozenset({"goal_diameter"}),
    "promo_squares": frozenset({"promo_squares"}),
    "promo_targets_flat": frozenset({"promo_targets"}),
    "promo_targets_index": frozenset({"promo_targets"}),
    "promo_optional": frozenset({"promo_optional"}),
    "promo_piece_types": frozenset({"promo_piece_types", "promo_flags"}),
    "promo_on_leaving": frozenset({"promo_leaving_flags"}),
    "entry_promotion": frozenset({"entry_promotion"}),
    "multi_royal": frozenset({"multi_royal"}),
}


# Every read site of every carrier, and why it is still a no-op under each refusal that
# leans on it. A key here that the walk no longer finds fails as loudly as a site the walk
# finds that is not here: the list is only worth reading if it is exact.
READERS: dict[str, dict[str, str]] = {
    # -- case 1, and case 13 where `en_passant` is declared beside a phantom promotion.
    "ep_type": {
        "game.rs::make": "sets the en passant window after the promoting piece's two-square "
        "move; `ep_passes` holds no entry for a piece that cannot double-step, and a piece "
        "that never stands never moves",
        "game.rs::generate_pseudo_legal_as": "`capturable_ep`, the three arms that take in "
        "passing; they need a window, which nothing under either refusal opens",
    },
    # -- case 3.
    "must_capture": {
        "game.rs::restrict": "the antichess filter; it keeps the captures when there are "
        "some, and with nothing takeable there are never any (Q118 closed en passant's hole)",
    },
    # -- case 4.
    "explosion": {
        "game.rs::make": "the blast, behind `captured != 0`, which no move reaches when "
        "nothing can be taken",
        "game.rs::leaves_royal_attacked": "the blast in the legality filter, behind an "
        "occupied target, and who is capturing; neither exists without a capture",
        "game.rs::generate_legal": "refuses the pin shortcut; the same verdict by the slower "
        "route, so it moves no answer",
        "eval.rs::of": "`Immunity::of` declines the immunity term under an explosion. "
        "**Evaluator, not rules**: with no capture the term's premise is gone anyway; "
        "re-read this site if case 4 is ever re-proven",
        "rounds.rs::make_round": "schema 26's blast, over the round's captures, of which "
        "there are none when nothing can be taken",
    },
    # -- cases 5 and 6.
    "drops": {
        "game.rs::initial": "the opening hands: empty, since the refusals need no stock",
        "game.rs::make": "the drop branch and the fill; the fill is behind `captured != 0` "
        "and `from_captures`, which the two refusals switch off between them",
        "game.rs::generate_pseudo_legal_as": "calls `generate_drops`, which returns at once on "
        "an empty hand; schema 18's placement gate (Q784) fires only on a non-empty one",
        "game.rs::royal_in_hand": "asks `royal_stocked` first (Q784), which needs a royal in "
        "the stock; the refusals need no stock, so it answers false",
        "game.rs::generate_drops": "walks the hand, which is empty for the life of the game",
        "rounds.rs::make_round": "schema 26's drop branch and fill, as `make`'s: the fill "
        "is behind a capture and `from_captures`",
        "game.rs::restrict": "the drop-mate filter; it filters drops, and there are none",
        "eval.rs::new": "`mean_non_royal` skips unlimited types; the refusals need no "
        "stock, so no type is unlimited and the mean is the undeclared one; also copies "
        "`has_drops`",
        "eval.rs::with_immunity": "the same `mean_non_royal`, the same answer",
        "eval.rs::of": "`Immunity::of` skips unlimited drop types, of which there are none",
        "features.rs::new": "copies `has_drops` for `HAND_MATERIAL`",
    },
    # Schema 26 (Q918a): a round mirrors `make`'s two readings.

    "eval.rs:has_drops": {
        "eval.rs::material": "adds the hand's value, which is zero with every hand empty",
        "eval.rs::immunity": "counts pieces in hand as present; there are none",
    },
    "features.rs:has_drops": {
        "features.rs::hand_material": "`HAND_MATERIAL`, zero with every hand empty",
    },
    # -- case 7.
    "bare_royal": {
        "game.rs::schema_two_ending": "the arm asks for a non-empty set of royals (Q134); "
        "with no piece marked royal no board satisfies it",
        "rounds.rs::round_ending_before_the_clocks": "schema 26's arm, the same test for "
        "each side",
        "tables.rs::has_v2_endings": "the gate on `schema_two_ending`, a disjunction that "
        "matches its arms one for one, so it switches on nothing else",
    },
    # -- cases 8 and 11.
    "check_limit": {
        "game.rs::make": "`delivered`; false at every ply with no royal (case 8) or with "
        "no move allowed to give check (case 11, which excludes an explosion)",
        "game.rs::make_fold": "schema 22: `make`'s `delivered` for a fold's own move, the same "
        "count under the same two cases",
        "game.rs::schema_two_ending": "the ending, on a count that never advances",
        "tables.rs::has_v2_endings": "the gate; see `bare_royal`",
        "tables.rs::has_goal_terms": "the evaluator's gate, a disjunction matching "
        "`goal_progress`'s terms one for one",
        "eval.rs::goal_progress": "the term, `clamp(0 - 0)` on a count that never advances",
        "features.rs::new": "copies the limit for `CHECK_MARGIN`",
        "features.rs::term": "`CHECK_MARGIN`, 0.0 on a count that never advances",
    },
    # -- cases 9 and 11.
    "forbid_check": {
        "game.rs::restrict": "guarded by `!t.royal_types.is_empty()` in the core itself "
        "(case 9); in case 11 it is the rule, not the refused one",
    },
    # -- case 10.
    "royal_facing_forbidden": {
        "game.rs::leaves_royal_attacked": "called only under `needs_check_filter()`, which "
        "requires a royal type",
        "game.rs::generate_legal": "refuses the pin shortcut, inside the same royal-only "
        "branch",
    },
    "facing_lines": {"tables.rs::facing_from": "the accessor `royals_face` reads"},
    "facing_paths": {"tables.rs::facing_path": "the accessor `royals_face` reads"},
    "facing_from()": {
        "game.rs::royals_face": "returns false the moment either royal square is negative",
    },
    "facing_path()": {"game.rs::royals_face": "behind the same royal-square guard"},
    # -- case 12.
    "castling": {
        "game.rs::generate_pseudo_legal_as": "only calls `generate_castling`, below",
        "game.rs::generate_castling": "needs an unmoved king and an unmoved rook on the "
        "king's own rank, and the `unmoved` mask is only ever cleared after ply zero",
        "game.rs::castling_partner": "answers a `MOVE_CASTLE`, which is never generated",
        "game.rs::castling_rook_squares": "answers a `MOVE_CASTLE`, which is never generated",
    },
    "castle_files": {
        "game.rs::generate_castling": "the same premise as `castling`",
        "game.rs::castling_partner": "answers a `MOVE_CASTLE`, which is never generated",
        "game.rs::castling_rook_squares": "answers a `MOVE_CASTLE`, which is never generated",
    },
    # -- case 13, the `reach` half: every reader is keyed on the piece standing somewhere.
    "reach": {
        "game.rs::schema_two_ending": "the ending: the named piece on a goal square",
        "tables.rs::has_v2_endings": "the gate; see `bare_royal`",
        "tables.rs::has_goal_terms": "the gate; see `check_limit`",
        "eval.rs::race_progress": "the race term, over pieces of the named type on the board",
        "eval.rs::immunity": "the immune racer's proximity, over pieces on the board",
        "eval.rs::goal_distance_of": "the distance of a piece of the named type that stands",
        "eval.rs::sealed_goals": "Q708's seal: goal squares held by an enemy piece, read only "
        "through `goal_distance_of`, over pieces of the named type, which never stand",
        "eval.rs::of": "`Seal::of` asks whether the race is priced; the seal it builds is "
        "read only as above",
        "features.rs::reach_progress": "`REACH_PROGRESS`, over pieces on the board",
        "game.rs::make": "schema 18f (Q786): whether the move took the named piece off a goal "
        "square, read only under `reach_leaves`, from the mover as it set out, so a piece "
        "that never stands never leaves",
        "rounds.rs::make_round": "schema 26's leaving test, `make`'s, per player",
        "rounds.rs::round_ending_before_the_clocks": "schema 26's arm: the named piece on a "
        "goal square, per side",
    },
    "goal_distance": {
        "eval.rs::goal_distance_of": "the distance of a piece that stands, for the race and "
        "for an immune racer",
        "features.rs::reach_progress": "the same, for `REACH_PROGRESS`",
    },
    "goal_diameter": {
        "eval.rs::race_progress": "a normaliser, multiplied by a distance nobody has",
        "eval.rs::nearest_to_goal": "the same normaliser, of a distance nobody has",
        "eval.rs::immunity": "the same normaliser, of a distance nobody has",
        "features.rs::reach_progress": "the same normaliser, of a distance nobody has",
    },
    "has_goal_terms()": {
        "eval.rs::score": "gates `goal_progress`, whose terms are the ones above",
    },
    "has_v2_endings()": {
        "game.rs::ending_before_the_clocks": "gates `schema_two_ending`, whose arms are the "
        "ones above",
        "rounds.rs::round_ending_before_the_clocks": "schema 26's gate on the same arms",
    },
    # -- case 13, the `promotion` half: keyed on a piece standing on, arriving on or
    # leaving a square, which a piece that can never exist does not do.
    "promo_squares": {"tables.rs::promotes_on": "the accessor; its callers are followed"},
    "promo_targets": {
        "tables.rs::promotion_targets": "the accessor; its callers are followed",
        "eval.rs::capture_threats": "the promotion closure `Immunity::of` and Q708's "
        "`Seal::of` read: what a type on the board can become. A phantom target widens a "
        "threat only if its own type stands to promote",
    },
    "promo_optional": {
        "tables.rs::promotion_optional": "the accessor; its callers are followed",
    },
    "promo_flags": {"tables.rs::is_promoting": "the accessor; its callers are followed"},
    "promo_leaving_flags": {"tables.rs::promotes_on_leaving": "the accessor; its callers are followed"},
    "promo_piece_types": {"features.rs::new": "copied as `promotable`"},
    "features.rs:promotable": {
        "features.rs::promotion_occupancy": "`PROMOTION_OCCUPANCY`, over pieces on the board",
    },
    "is_promoting()": {
        "game.rs::generate_pseudo_legal_as": "asked of the moving piece",
        "game.rs::push_transits": "asked of the moving piece (schema 19)",
        "game.rs::push_portal_moves": "asked of each piece before its transits (schema 19)",
        "game.rs::push_through": "asked of the moving piece going through a fold (schema 24)",
        "features.rs::promotion_occupancy": "asked of a piece on the board",
    },
    "promotes_on()": {
        "game.rs::generate_pseudo_legal_as": "asked of a moving piece's origin",
        "game.rs::push_arrival": "asked of a moving piece's target",
        "game.rs::push_transits": "asked of a moving piece's exit (schema 19)",
        "game.rs::push_portal_moves": "asked of a moving piece's origin (schema 19)",
        "features.rs::promotion_occupancy": "asked of a piece on the board",
    },
    "promotes_on_leaving()": {
        "game.rs::generate_pseudo_legal_as": "asked of the moving piece",
        "game.rs::push_portal_moves": "asked of each piece before its transits (schema 19)",
    },
    "promotion_optional()": {
        "game.rs::generate_pseudo_legal_as": "asked of the moving piece",
        "game.rs::push_transits": "asked of the moving piece (schema 19)",
        "game.rs::push_through": "asked of the moving piece going through a fold (schema 24)",
    },
    "promotion_targets()": {
        "game.rs::generate_pseudo_legal_as": "asked of the moving piece",
        "game.rs::push_arrival": "asked of the moving piece",
        "game.rs::push_transits": "asked of the moving piece (schema 19)",
    },
    "entry_promotion": {"tables.rs::arriving_value": "the accessor; its callers are followed"},
    "arriving_value()": {
        "game.rs::make": "the value of a piece entering play, by drop or by strike",
        "game.rs::strike_writes": "the value a struck piece becomes where it lands",
        "game.rs::play_move_for_filter": "the same write in the legality filter",
        "game.rs::restrict": "the immobility filter, asking of a dropped piece as it "
        "stands after arrival",
        "game.rs::is_forbidden_drop_mate": "the dropped piece's value",
        "rounds.rs::plan": "schema 26: a dropped piece's value, as `make`'s drop branch",
        "rounds.rs::make_round": "schema 26: an own merge's mover, as `strike_writes`",
        "rounds.rs::contact_in_place": "schema 26: a contact's two pieces, as `strike_writes`",
        "rounds.rs::strike_after": "schema 26: a struck piece and its captor, as "
        "`strike_writes`",
    },
    "rounds.rs:mover": {
        "rounds.rs::make_round": "schema 26: the dropped piece's value `plan` read, placed "
        "as `make` places it",
    },
    # `multi_royal` moves only where the spec declares a real royal promotion beside the
    # phantom one; every reader counts royals standing on the board.
    "multi_royal": {
        "game.rs::royal_values": "which values are royal, for the royal hint",
        "game.rs::royal_square": "where a side's royal stands",
        "game.rs::royal_count": "how many royals a side has standing",
        "game.rs::in_check_at": "whether any standing royal is attacked",
        "game.rs::compute_in_check": "the same, memoised",
        "game.rs::make": "`delivered`, over standing royals",
        "game.rs::make_fold": "schema 22: the same `delivered` after a fold's own move",
        "game.rs::leaves_royal_attacked": "the legality filter, over standing royals",
        "game.rs::generate_legal": "the pin shortcut's single-royal premise",
        "game.rs::restrict": "`forbid_check`'s filter, over standing royals",
        "game.rs::is_forbidden_drop_mate": "whether a drop checks a standing royal",
    },
}


_FN = re.compile(
    r"^(\s*)(?:pub(?:\([a-z]+\))?\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+([a-z_0-9]+)\s*[(<]"
)


def _rust_lines(path: Path) -> list[tuple[str, str]]:
    """Each line of a Rust file as ``(enclosing function, code)``, comments stripped.

    Comments go first for the model test's reason: `game.rs` *discusses* these rules at
    length, and a guard that counted its own explanation would fail on a rewording. The
    function is tracked from its opening line to the closing brace at its indentation,
    which is how `rustfmt` lays out every function in `rust/src/`; a one-line body closes
    on its own line. Outside any function the name is ``<item>`` -- a struct field or a
    constant, which reads nothing.
    """
    out: list[tuple[str, str]] = []
    current: str | None = None
    indent = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.split("//")[0]
        opened = _FN.match(line) if current is None else None
        if opened:
            current, indent = opened.group(2), opened.group(1)
        out.append((current or "<item>", code))
        one_line = bool(opened) and 0 < code.count("{") == code.count("}")
        if current is not None and (line.rstrip() == indent + "}" or one_line):
            current = None
    return out


def _core_lines() -> dict[str, list[tuple[str, str]]]:
    return {path.name: _rust_lines(path) for path in sorted(CORE.glob("*.rs"))}


def _field(name: str) -> re.Pattern[str]:
    """A field read, ``.name`` not followed by a call."""
    return re.compile(r"\." + name + r"\b(?!\s*\()")


def read_sites() -> dict[str, set[str]]:
    """Every carrier of a refused key, and the ``file::function`` sites that read it.

    A carrier is a `Tables` member (``name``), a `Tables` method that reads one
    (``name()``, its callers followed to a fixed point), or a struct-literal copy of one
    into another struct's field (``file:name``, its readers in that file followed).
    """
    lines = _core_lines()
    tables_fns = {fn for fn, _ in lines[TABLES.name]} - {"<item>"}
    members = sorted(set().union(*MEMBERS.values()))
    patterns: dict[str, tuple[re.Pattern[str], str | None]] = {
        member: (_field(member), None) for member in members
    }
    found: dict[str, set[str]] = {}
    queue = list(patterns)
    while queue:
        carrier = queue.pop()
        pattern, only_in = patterns[carrier]
        for name, rows in lines.items():
            if only_in is not None and name != only_in:
                continue
            for fn, code in rows:
                if not pattern.search(code):
                    continue
                found.setdefault(carrier, set()).add(f"{name}::{fn}")
                if name == TABLES.name and fn in tables_fns and f"{fn}()" not in patterns:
                    patterns[f"{fn}()"] = (
                        re.compile(r"(?:\.|\bSelf::|\bTables::)" + fn + r"\s*\("), None
                    )
                    queue.append(f"{fn}()")
                copy = re.match(r"^\s*([a-z_0-9]+):\s*\S", code)
                if copy and copy.group(1) not in members:
                    key = f"{name}:{copy.group(1)}"
                    if key not in patterns:
                        patterns[key] = (_field(copy.group(1)), name)
                        queue.append(key)
    return found


def _cases(carrier: str) -> str:
    """The refusals a member carries, for the failure message."""
    keys = {key for key, members in MEMBERS.items() if carrier in members}
    cases = sorted(case for case, (_, fields) in CLAUSES.items() if keys & fields)
    if not cases:
        return "an accessor or a copy of a member; see READERS for whose"
    return "cases " + ", ".join(str(case) for case in cases)


class TestEveryReaderOfARefusedFieldIsNamed:
    """The core half: every read of a field an inert-rule refusal stands on is on `READERS`."""

    def test_no_unlisted_site_reads_a_refused_field(self) -> None:
        found = read_sites()
        assert found, "the walk matched nothing at all -- the test has gone inert"
        unlisted = sorted(
            f"{site} reads {carrier} ({_cases(carrier)})"
            for carrier, sites in found.items()
            for site in sites - set(READERS.get(carrier, {}))
        )
        assert not unlisted, (
            "a site in rust/src/ reads a field that an inert-rule refusal in "
            "spec/schema.py::_check_inert_rules is proven against, and it is not on "
            "READERS: " + "; ".join(unlisted) + ". A new reader un-proves the refusal "
            "silently. Re-run step 3 of docs/reference/bugs-investigations.md, 'What "
            "proving a rule inert requires', for each case named: if the site is the rule "
            "itself and a no-op under the refused spelling, add it to READERS saying why; "
            "if it is anything else, the refusal is refusing two rules under one name."
        )

    def test_the_list_has_not_gone_stale_the_other_way(self) -> None:
        """A site that stopped reading a field leaves the list, as the model test's does."""
        found = read_sites()
        listed = {carrier: set(sites) for carrier, sites in READERS.items()}
        assert found == listed

    def test_every_listed_site_says_why(self) -> None:
        for carrier, sites in READERS.items():
            for site, why in sites.items():
                assert len(why.split()) >= 3, (carrier, site)

    def test_every_member_is_still_read_somewhere(self) -> None:
        """A member no site reads means a rename the pattern cannot see, not a dead field."""
        found = read_sites()
        silent = sorted(m for m in set().union(*MEMBERS.values()) if not found.get(m))
        assert not silent, silent

    def test_every_refused_key_has_a_member(self) -> None:
        assert set(MEMBERS) == REFUSED_KEYS


class TestTheBoundaryReadsThemOnce:
    """Each refused key crosses into `Tables` in `from_py` and is never looked up again."""

    @staticmethod
    def _const(key: str) -> str:
        return key.upper()

    def test_each_key_is_declared_in_the_vocabulary(self) -> None:
        source = TABLES.read_text(encoding="utf-8")
        for key in REFUSED_KEYS:
            assert f'{self._const(key)} => "{key}"' in source, key

    def test_each_key_is_read_only_by_from_py(self) -> None:
        stray = []
        read = set()
        for name, rows in _core_lines().items():
            for fn, code in rows:
                for key in REFUSED_KEYS:
                    if not re.search(r"\b" + self._const(key) + r"\b", code):
                        continue
                    site = f"{name}::{fn}"
                    if site == f"{TABLES.name}::from_py":
                        read.add(key)
                    elif site != f"{TABLES.name}::<item>":
                        stray.append(f"{site} names keys::{self._const(key)}")
        assert not stray, stray
        assert read == REFUSED_KEYS, sorted(REFUSED_KEYS - read)


# The `CompiledSpec` attributes the bridge flattens into those keys.
_SPEC_ATTRIBUTES = frozenset({
    "en_passant", "promotions", "must_capture", "explosion", "drops", "bare_royal",
    "check_limit", "forbid_check", "royal_facing_forbidden", "facing_lines", "castling",
    "reach", "goal_distance", "goal_diameter", "entry_promotion", "multi_royal",
})

# Where `bridge.py` may read them, and why each is still only a flattening.
BRIDGE_READERS: dict[str, str] = {
    "build_tables": "the flattening itself, where each refused key is written",
    "rust_supports": "a support check whose every branch sits behind a `_CORE_KNOWS_*` "
    "constant that is True; it sends nothing across the boundary",
}

# Where `bridge.py` may spell the refused keys, and why.
BRIDGE_SPELLERS: dict[str, str] = {
    "build_tables": "writes them",
    "_INERT_DEFAULTS": "the fingerprint's default values; read by `canonical_tables` for "
    "every key alike, never for one rule",
    "_ALWAYS_MEANINGFUL": "the fingerprint's other half, a set of names",
}


def _bridge_definitions() -> list[tuple[str, ast.AST]]:
    named: list[tuple[str, ast.AST]] = []
    for node in ast.parse(BRIDGE.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            named.append((node.name, node))
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            named.append((getattr(target, "id", "<assignment>"), node))
        else:
            named.append(("<module>", node))
    return named


class TestTheBridgeOnlyFlattensThem:
    """The Python half: the bridge reads these rules in `build_tables` and spells them there."""

    def test_the_spec_attributes_are_read_only_where_listed(self) -> None:
        readers = {
            name
            for name, node in _bridge_definitions()
            for sub in ast.walk(node)
            if isinstance(sub, ast.Attribute)
            and sub.attr in _SPEC_ATTRIBUTES
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "spec"
        }
        assert readers == set(BRIDGE_READERS), (
            "bridge.py reads a rule an inert-rule refusal stands on somewhere new: "
            f"{sorted(readers - set(BRIDGE_READERS))}. A second boundary field derived "
            "from a refused rule is a new reader of it; name it in BRIDGE_READERS and "
            "re-prove the clauses it serves"
        )

    def test_the_keys_are_spelled_only_where_listed(self) -> None:
        spellers = {
            name
            for name, node in _bridge_definitions()
            for sub in ast.walk(node)
            if isinstance(sub, ast.Constant) and sub.value in REFUSED_KEYS
        }
        assert spellers == set(BRIDGE_SPELLERS), sorted(spellers ^ set(BRIDGE_SPELLERS))
