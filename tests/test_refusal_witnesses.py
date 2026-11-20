"""A witness for every refusal, and what the search offers for it (R-L3, B5a and B5b).

`docs/reference/refusal-offers.md` §9 row B5 and §5's measured table. The witnesses are data
(`tests/refusal_witnesses/`, read by `tests/refusal_witness.py`); this file holds that every
refusal of the covered families (`refusal_witness.FAMILIES`) has one, that each is refused with
its entry and with nothing new (§8 test 6), and, in the slow tier, that every offer the search
makes for them is allowed and that the witnesses with no offer may only become fewer.

A lift deletes its `refusals.toml` entry and, now, its witness (the stale witness fails here). A
new refusal of a covered family needs a witness in the same commit, or fails here too.
"""

from __future__ import annotations

import json

import pytest
from refusal_witness import (
    FAMILIES,
    STORE,
    covered,
    entries,
    find,
    made,
    met,
    stored,
    unwritable,
)
from test_instead import _assert_offer

from chess_variant_sandbox.spec.instead import OFFERS, VALIDATIONS

# The entries no witness can be written for (`refusal-offers.md` §5, the as-built notes): B5a's
# two, refused with no sentence `entry_of` resolves to them, and B5b's one, which no spec reaches
# (a portal end's type is never a declared piece). A list that may only shrink.
UNWRITABLE = {"K7.facing", "K7.spy", "shared.portal_end"}

# The witnesses the search finds no offer for (§4.3: no offer is an answer): a ceiling that
# may only fall. Measured 2026-11-20 by `uv run python tests/refusal_witness.py`: B5a's four,
# then B5b's thirteen (§5.3: a refusal several sites hold, each left out alone mending nothing).
NO_OFFER = {
    "K11", "X3", "X4", "X13.sides",
    "push.W4", "push.S1", "push.N4", "pass.P8", "parts.ramp",
    "run.wrap", "run.portals", "run.rounds", "run.sweep", "run.npcs", "run.duck", "run.sight",
    "run.rest",
}

_STORED = sorted(rid for rid, witness in stored().items() if witness.b2 is None)


def test_every_family_is_covered() -> None:
    """With B5b in, `FAMILIES` is every family of `refusals.toml`, so a refusal in any family,
    a new family's included, fails the next test unless it has a witness or a listed reason."""
    assert {raw["family"] for raw in entries().values()} - set(FAMILIES) == set()


def test_every_refusal_of_the_covered_families_has_a_witness() -> None:
    need = set(covered())
    have = set(stored()) | set(unwritable())
    assert need - have == set(), "a refusal with no witness: write one in tests/refusal_witnesses/"
    assert set(stored()) & set(unwritable()) == set()


def test_no_witness_is_for_a_refusal_that_is_gone() -> None:
    """A lift deletes the entry; its witness goes with it."""
    known = entries()
    for path in sorted(STORE.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["family"] in FAMILIES, path.name
        for rid in [*data.get("witnesses", {}), *data.get("none", {})]:
            assert rid in known, f"{path.name}: {rid} is no refusal of refusals.toml"
            assert known[rid]["family"] == data["family"], (path.name, rid)
            assert known[rid]["reason"] != "inert", (path.name, rid)


def test_the_unwritable_list_may_only_shrink() -> None:
    assert set(unwritable()) <= UNWRITABLE
    assert all(why.strip() for why in unwritable().values())


@pytest.mark.parametrize("rid", _STORED)
def test_each_witness_is_refused_with_its_entry(rid: str) -> None:
    """§8 test 6 for the witnesses: what the validator says of each is pinned, as entries."""
    witness = stored()[rid]
    trial, wanted, _base = made(witness)
    said = met(trial)
    assert rid in said
    assert witness.meets is not None and said == list(witness.meets)
    if wanted is not None:
        assert wanted.startswith(("rules.", "pieces[", "board.", "setup."))


@pytest.mark.slow
def test_every_offer_for_a_witness_is_allowed_and_no_offer_may_only_fall() -> None:
    """§8 test 1 over every witness, and the ceiling on *no offer*. About 45 s with B5b's, 17 s
    of it on the sweep limits (`hold.C2`, `hold.C3`), whose validator runs ~0.2 s a draft."""
    none = set()
    for rid in covered():
        if rid not in stored():
            continue
        found = find(rid)
        trial, wanted, base = made(stored()[rid])
        assert found.search.validations <= VALIDATIONS
        assert len(found.search.offers) <= OFFERS
        for offer in found.search.offers:
            _assert_offer(trial, offer, wanted, base)
        if not found.search.offers:
            none.add(rid)
    assert none <= NO_OFFER, none - NO_OFFER
