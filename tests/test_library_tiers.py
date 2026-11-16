"""The library's tiers against chess, in place of a three-decimal order (R-S6c).

`docs/reference/v14-registration.md` §2.3 fixed the rule before the page was built: a
game's tier is its relation to chess, the calibration baseline, by `bootstrap.clears`
alone (*above chess*, *cannot be told from chess*, *below chess*), no order inside a tier,
no new threshold. Chess is in these tests because the rule is defined against it, the
one game every score is measured against; nothing here asserts a fact about chess's rules.

Held down here:

- **The rule** (`library.tier_of`) on fixture scores: a game that clears chess, one chess
  clears, one neither does, chess itself, a game with no current score of its own (a
  visitor's one-run number rides beside the card and is no score of the game's), a
  provisional estimate, a refused game, a game measured without a verdict, an interval
  built on sweeps, and a chess with no current score.
- **The listing** in both states: every score provisional (today, between the switch and
  the replay) puts every game in *not yet measured* and none in a tier; every score
  current fills the tiers with no other change.
- **The site's `?sort=index`** orders by tier, then name, never by the number.
- **The page** orders by tier and name, draws the tiers as groups with their words, and
  turns only the three tiers round when the direction flips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from test_provisional_estimate import EARLIER, _measure
from test_web import _in_page

from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.site import browse
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.server import App


def _score(total: float, half: float = 0.02, **extra: Any) -> dict[str, Any]:
    """A current score under the current weights, with an interval of +/- `half`."""
    return {"status": "current", "weights_version": DEFAULT_WEIGHTS, "total": total,
            "viable": True, "interval": {"ci_low": total - half, "ci_high": total + half,
                                         "deterministic_pairings": 0.0}, **extra}


CHESS = _score(0.90)


class TestTheRule:
    def test_a_game_that_clears_chess_is_above(self) -> None:
        assert library.tier_of(_score(0.97), CHESS) == library.ABOVE

    def test_a_game_chess_clears_is_below(self) -> None:
        assert library.tier_of(_score(0.70), CHESS) == library.BELOW

    def test_a_game_neither_clears_cannot_be_told_from_chess(self) -> None:
        # Higher, but the intervals overlap: not *above*, whatever the third decimal says.
        assert library.tier_of(_score(0.93), CHESS) == library.LEVEL
        assert library.tier_of(_score(0.87), CHESS) == library.LEVEL

    def test_chess_itself_is_in_the_middle_tier(self) -> None:
        assert library.tier_of(CHESS, CHESS, is_chess=True) == library.LEVEL

    def test_no_new_threshold_the_intervals_alone_decide(self) -> None:
        """Touching intervals do not clear (`clears` is strict); a hair apart, they do."""
        touching = _score(0.94, half=0.02)  # low end 0.92 == chess's high end
        assert library.tier_of(touching, CHESS) == library.LEVEL
        apart = _score(0.9401, half=0.02)
        assert library.tier_of(apart, CHESS) == library.ABOVE

    def test_a_game_with_no_score_of_its_own_is_not_yet_measured(self) -> None:
        """A visitor's one-run number from their own machine is not the game's score: the
        game's own is `None`, and it is in no tier."""
        assert library.tier_of(None, CHESS) == library.UNMEASURED

    def test_a_one_run_current_score_is_tiered_by_its_own_interval(self) -> None:
        one_run = _score(0.60, half=0.05, seeds=[0])
        assert library.tier_of(one_run, CHESS) == library.BELOW

    def test_a_provisional_score_is_not_yet_measured_whatever_its_number(self) -> None:
        provisional = {**_score(0.99), "status": "provisional"}
        assert library.tier_of(provisional, CHESS) == library.UNMEASURED
        earlier = {**_score(0.99), "weights_version": EARLIER}
        assert library.tier_of(earlier, CHESS) == library.UNMEASURED

    def test_a_refused_game_is_outside_the_tiers(self) -> None:
        assert library.tier_of(_score(0.99, viable=False), CHESS) == library.REFUSED

    def test_a_game_without_a_verdict_or_an_interval_is_not_placed(self) -> None:
        unread = _score(0.99, viable=False, measured=False)
        assert library.tier_of(unread, CHESS) == library.UNPLACED
        bare = {"status": "current", "weights_version": DEFAULT_WEIGHTS, "total": 0.99}
        assert library.tier_of(bare, CHESS) == library.UNPLACED

    def test_an_interval_built_on_sweeps_separates_nothing(self) -> None:
        swept = _score(0.99)
        swept["interval"]["deterministic_pairings"] = 1.0
        assert library.tier_of(swept, CHESS) == library.LEVEL

    def test_a_chess_with_no_current_score_places_nobody(self) -> None:
        stale_chess = {**CHESS, "status": "provisional"}
        assert library.tier_of(_score(0.99), stale_chess) == library.UNPLACED
        assert library.tier_of(_score(0.99), None) == library.UNPLACED

    def test_every_group_has_its_words_and_the_tiers_lead(self) -> None:
        legend = library.tier_legend()
        assert legend["order"][:3] == legend["tiers"] == [
            library.ABOVE, library.LEVEL, library.BELOW]
        assert set(legend["groups"]) == set(legend["order"])
        assert all(words["title"] and words["note"] for words in legend["groups"].values())
        # Plain words for a visitor: no internal names in the prose.
        prose = json.dumps(legend)
        for internal in ("clears", "bootstrap", "v13", "v14", "weights", "likeness"):
            assert internal not in prose


SCORES = {"chess": 0.90, "gardner": 0.97, "losalamos": 0.93, "dull": 0.40}


def _listing(root: Path, weights_version: str | None = None) -> dict[str, Any]:
    extra = {} if weights_version is None else {"weights_version": weights_version}
    for name, index in SCORES.items():
        _measure(root, name, index, **extra)
    app = App(root / "rulesets", root / "runs", archive_dir=root / "archive")
    payload: dict[str, Any] = app.get("/api/library")
    return payload


def _by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {card["name"].replace(" ", ""): card for card in payload["games"]}


class TestTheListing:
    def test_every_score_provisional_puts_every_game_outside_the_tiers(
        self, tmp_path: Path
    ) -> None:
        """The state between the switch and the replay: every stored score is under the
        earlier version, so none is compared, and each keeps its number, marked."""
        cards = _by_name(_listing(tmp_path, weights_version=EARLIER))
        assert {card["tier"] for card in cards.values()} == {library.UNMEASURED}
        assert all(card["status"] == "provisional" and card["index"] is not None
                   for card in cards.values())

    def test_every_score_current_fills_the_tiers(self, tmp_path: Path) -> None:
        payload = _listing(tmp_path)
        cards = _by_name(payload)
        # `_measure`'s interval is +/- 0.02: gardner clears chess, losalamos does not.
        assert {name: card["tier"] for name, card in cards.items()} == {
            "chess": library.LEVEL, "gardner": library.ABOVE,
            "losalamos": library.LEVEL, "dull": library.BELOW,
        }
        assert [name for name, card in cards.items() if card["tierChess"]] == ["chess"]
        # Each game's own number stays on its card.
        assert {name: card["index"] for name, card in cards.items()} == SCORES
        assert payload["tiers"] == library.tier_legend()

    def test_the_site_orders_by_tier_then_name_never_by_the_number(
        self, tmp_path: Path
    ) -> None:
        cards = _listing(tmp_path)["games"]
        found = [card["name"].replace(" ", "") for card in browse.order(cards, browse.INDEX)]
        # chess and losalamos share a tier, so they go by name, not by 0.90 and 0.93.
        assert found == ["gardner", "chess", "losalamos", "dull"]


_TIERED = "TIERS = " + json.dumps(library.tier_legend()) + """;
LIBRARY = [
  { name: 'zeta', hash: 'z', tier: 'level', index: 0.95 },
  { name: 'alpha', hash: 'a', tier: 'level', index: 0.88 },
  { name: 'top', hash: 't', tier: 'above', index: 0.99 },
  { name: 'low', hash: 'l', tier: 'below', index: 0.50 },
  { name: 'old', hash: 'o', tier: 'unmeasured', index: 0.97, status: 'provisional' },
  { name: 'nope', hash: 'n', tier: 'refused', index: 0.60 },
];
BROWSE.sort = 'index';
BROWSE.group = 'none';
"""


class TestThePage:
    def test_it_orders_by_tier_then_name(self) -> None:
        found = _in_page(_TIERED + "BROWSE.descending = true;",
                         "ordered(LIBRARY).map(g => g.name)")
        assert found == ["top", "alpha", "zeta", "low", "nope", "old"]

    def test_turning_the_direction_round_turns_only_the_tiers(self) -> None:
        found = _in_page(_TIERED + "BROWSE.descending = false;",
                         "ordered(LIBRARY).map(g => g.name)")
        assert found == ["low", "alpha", "zeta", "top", "nope", "old"]

    def test_the_tiers_are_the_groups_with_their_words(self) -> None:
        found = _in_page(_TIERED + "BROWSE.descending = true;",
                         "grouped(ordered(LIBRARY)).map(g => [g.title, g.games.length])")
        words = library.TIER_WORDS
        assert found == [[words[library.ABOVE]["title"], 1],
                         [words[library.LEVEL]["title"], 2],
                         [words[library.BELOW]["title"], 1],
                         [words[library.REFUSED]["title"], 1],
                         [words[library.UNMEASURED]["title"], 1]]

    def test_another_ordering_still_reads_the_number(self) -> None:
        found = _in_page(_TIERED + "BROWSE.sort = 'name'; BROWSE.descending = false;",
                         "grouped(ordered(LIBRARY)).length")
        assert found == 1
