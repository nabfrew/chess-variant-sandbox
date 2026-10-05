"""A score is drawn from the version it was taken under, never from a count in the text.

`docs/archive/plan-variety.md` §5.4, box A2. The user took rule yield, the whole elegance family,
out of the score from `v11` on. So the pages stop saying *sixteen things measured* and
read the count off the score's own version, the browse menu offers only the default
version's parts, and the designer is neither told about few rules nor rewarded for them.

A six-family version is built here from `v10` without its elegance family, which was
written before `v11` existed and still isolates the family's removal from everything else
`v11` changed; since C2 (Q391) the shipped `v11` is drawn beside it.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import node_session
import pytest

from chess_variant_sandbox.metrics import scoring
from chess_variant_sandbox.web import library

SIX = "vsix"


@pytest.fixture
def six_families(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A weights version that is `v10` with the elegance family taken out.

    Written beside copies of every shipped version, so `v10` still loads from the same
    directory and the two can be drawn side by side.
    """
    folder = tmp_path / "weights"
    folder.mkdir()
    for path in scoring.WEIGHTS_DIR.glob("*.json"):
        shutil.copy(path, folder / path.name)
    data = json.loads((scoring.WEIGHTS_DIR / "v10.json").read_text(encoding="utf-8"))
    data["version"] = SIX
    data["components"] = {
        name: entry
        for name, entry in data["components"].items()
        if entry.get("family") != "elegance"
    }
    (folder / f"{SIX}.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(scoring, "WEIGHTS_DIR", folder)
    return SIX


def _score(version: str) -> dict[str, Any]:
    """A stored score's shape, worked by the scoring module under `version`."""
    weights = scoring.load_weights(version)
    scores = {name: 0.5 for name in weights.components()}
    total, contributions = weights.combine(scores)
    return {
        "weights_version": version,
        "total": total,
        "viable": True,
        "violations": [],
        "families": weights.family_scores(scores),
        "scores": scores,
        "contributions": contributions,
        "raw": {name: 1.0 for name in scores},
        "seeds": [0, 1, 2, 3],
        "games": 400,
    }


def _drawn(pairs: list[tuple[dict, dict | None]]) -> list[str]:
    """What `scoreHtml` draws for each (score, guide), from one run of the page in node."""
    from page_text import PAGE

    script = re.search(r"<script>(.*)</script>", PAGE, re.S)
    assert script is not None
    stub = """
      const el = new Proxy({}, { get: (t, k) => (k === 'style' ? {} : () => null),
                                 set: () => true });
      global.document = { getElementById: () => el, querySelector: () => null,
                          querySelectorAll: () => [], addEventListener: () => {},
                          createElement: () => el, body: el };
      global.window = { matchMedia: () => ({ matches: false, addEventListener: () => {} }),
                        addEventListener: () => {}, scrollTo: () => {}, innerWidth: 1200 };
      global.location = { pathname: '/game', hash: '' };
      global.fetch = () => new Promise(() => {});
    """
    shown = (
        f"JSON.stringify({json.dumps(pairs)}.map(([s, g]) => scoreHtml(s, g, null)))"
    )
    done = node_session.run(stub + script.group(1) + "\nconsole.log(" + shown + ");")
    assert done.returncode == 0, done.stderr
    return list(json.loads(done.stdout))


def _bars(html: str) -> list[str]:
    # A family's name is a tap target since W3 (G27), so its span carries more than a title.
    return re.findall(r'<div class="fam">\s*<span [^>]*title="[^"]*">([^<]*)</span>', html)


class TestTheScoreNamesItsOwnParts:
    def test_six_families_draw_six_bars_and_count_their_own_components(
        self, six_families: str
    ) -> None:
        """The new path: a version without elegance draws six bars, and every count on
        the panel is that version's -- twelve components, five of them on a band."""
        guide = library.score_guide({"weights_version": six_families})
        assert guide is not None and "elegance" not in guide["families"]
        [html] = _drawn([(_score(six_families), guide)])

        assert sorted(_bars(html)) == [
            "balance", "decisions", "depth", "drama", "pacing", "variety",
        ]
        assert "<h3>The twelve things measured</h3>" in html
        assert "one of the twelve things measured" in " ".join(html.split())
        assert html.count('<details class="comp">') == 12
        assert "Five of the twelve have a band" in html
        assert "rule yield" not in html and "elegance" not in html
        assert "sixteen" not in html and "thirteen" not in html

    def test_a_v11_score_draws_six_bars_and_eleven_parts(self) -> None:
        """The shipped six-family version, drawn from its own guide: six bars, eleven
        components, the same five of them on a band as `v10`, and no rule yield."""
        guide = library.score_guide({"weights_version": "v11"})
        [html] = _drawn([(_score("v11"), guide)])

        assert sorted(_bars(html)) == [
            "balance", "decisions", "depth", "drama", "pacing", "variety",
        ]
        assert "<h3>The eleven things measured</h3>" in html
        assert html.count('<details class="comp">') == 11
        assert "Five of the eleven have a band" in html
        assert "rule yield" not in html and "elegance" not in html

    def test_a_v10_score_renders_as_it_did(self) -> None:
        """`v10` is still a live version, no longer the default. Its score keeps its seven
        bars, elegance and rule yield among them, and the counts are `v10`'s own: the
        text used to say *sixteen*, which was `v9`'s count and wrong for `v10`."""
        guide = library.score_guide({"weights_version": "v10"})
        [html] = _drawn([(_score("v10"), guide)])

        assert sorted(_bars(html)) == [
            "balance", "decisions", "depth", "drama", "elegance", "pacing", "variety",
        ]
        assert "<h3>The thirteen things measured</h3>" in html
        assert html.count('<details class="comp">') == 13
        assert "rule yield" in html
        assert "Five of the thirteen have a band" in html
        assert "sixteen" not in html

    def test_a_score_whose_version_is_gone_counts_what_it_carries(self) -> None:
        """No guide is a version this checkout no longer ships. The count is the score's
        own components, and no band is claimed, because nothing says which are bands."""
        score = _score("v10")
        [html] = _drawn([(score, None)])
        assert "<h3>The thirteen things measured</h3>" in html
        assert "have a band" not in html and "has a band" not in html

    def test_no_text_on_the_page_states_how_many_things_are_measured(self) -> None:
        """The vocabulary (i)s are shared by every version, so they carry no count."""
        from page_text import PAGE

        for said in ("sixteen measurements", "sixteen things", "six of the sixteen",
                     "thirteen measurements", "thirteen things"):
            assert said not in PAGE


class TestTheBrowseMenuOffersTheDefaultVersionsParts:
    def test_elegance_leaves_the_menu_when_a_version_without_it_is_the_default(
        self, six_families: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(library, "DEFAULT_WEIGHTS", six_families)
        legend = library.metric_legend()
        assert legend["version"] == six_families
        assert len(legend["families"]) == 6
        assert "elegance" not in legend["families"]
        assert "rule_yield" not in legend["components"]
        assert len(legend["components"]) == 12

    def test_under_v12_the_menu_offers_v11s_parts(self) -> None:
        """`v12` (EV4) is `v11`'s design with its ends given sources, so the menu is still
        `v11`'s six families and eleven components. Before `v11` (C2, Q391) this read
        `v10`'s menu, elegance and all."""
        legend = library.metric_legend()
        assert legend["version"] == scoring.DEFAULT_WEIGHTS == "v12"
        assert len(legend["families"]) == 6
        assert "elegance" not in legend["families"]
        assert "rule_yield" not in legend["components"]
        assert "position_divergence" in legend["components"]
        assert "opening_diversity" not in legend["components"]
        assert len(legend["components"]) == 11

    def test_the_glossary_still_explains_rule_yield_on_an_older_score(
        self, six_families: str
    ) -> None:
        """`v10` stays live, so its score still has every part explained."""
        from chess_variant_sandbox.metrics.glossary import FAMILIES, MEASUREMENTS

        assert "elegance" in FAMILIES and "rule_yield" in MEASUREMENTS
        guide = library.score_guide({"weights_version": "v10"})
        assert guide is not None
        assert guide["components"]["rule_yield"]["what"]
        assert guide["families"]["elegance"]["means"]


class TestTheDesignerIsNotToldAboutFewRules:
    def test_feedback_says_nothing_about_a_small_or_a_poor_yielding_rulebook(self) -> None:
        from chess_variant_sandbox.llm.feedback import describe
        from chess_variant_sandbox.metrics import load_weights, score_components
        from chess_variant_sandbox.metrics.components import Components

        weights = load_weights("v10")
        raw = {name: 0.5 for name in weights.components()}
        raw.update({"mean_plies": 40, "mean_branching": 20, "decisive_rate": 0.6,
                    "skill_gradient": 50, "search_advantage": 0.9,
                    "rule_mass": 20, "rule_yield": 1.0})
        result = score_components(
            "h", "g", Components(raw=raw, diagnostics={"games": 100}), weights
        )
        text = describe(result).lower()
        assert "compact" not in text and "explain quickly" not in text
        assert "a lot of rule here" not in text and "could be removed" not in text

    def test_feedback_says_nothing_about_a_large_rulebook_either(self) -> None:
        """The other end (Q400). "The rules are heavy" fired past a mass of 110, which is
        the ceiling the user declined, said as advice. No observation reads rule mass or
        rule yield at all, so neither end can come back as a new sentence."""
        from chess_variant_sandbox.llm.feedback import _OBSERVATIONS, _STRENGTHS, describe
        from chess_variant_sandbox.metrics import load_weights, score_components
        from chess_variant_sandbox.metrics.components import Components

        read = {component for component, _, _ in _OBSERVATIONS + _STRENGTHS}
        assert "rule_mass" not in read and "rule_yield" not in read

        weights = load_weights("v10")
        raw = {name: 0.5 for name in weights.components()}
        raw.update({"mean_plies": 40, "mean_branching": 20, "decisive_rate": 0.6,
                    "skill_gradient": 50, "search_advantage": 0.9,
                    "rule_mass": 400, "rule_yield": 0.1})
        result = score_components(
            "h", "g", Components(raw=raw, diagnostics={"games": 100}), weights
        )
        text = describe(result).lower()
        assert "heavy" not in text and "long explanation" not in text

    def test_the_brief_does_not_ask_for_few_rules(self) -> None:
        """The brief's *Elegance* bullet was the family in prose (Q400). What is left asks
        that every piece and rule matter to play, which the remaining score reads through
        how much each piece is used, and says nothing about how many there are."""
        from chess_variant_sandbox.llm.propose import DESIGN_BRIEF, SCHEMA_PREAMBLE

        shown = " ".join((DESIGN_BRIEF + " " + SCHEMA_PREAMBLE).split()).lower()
        for said in ("elegance", "few pieces", "few exceptions", "fit in a paragraph",
                     "special cases are expensive", "should have none", "has to earn",
                     "earns its place"):
            assert said not in shown, said

        brief = " ".join(DESIGN_BRIEF.split()).lower()
        assert "**purpose.** every piece and every rule should change what a player" in brief
        # Kept: it is about which fix is the better one, not how many rules there are.
        assert "a special rule to patch a problem that a change to a piece would fix" in brief

    def test_the_weakest_family_advice_never_says_cut_the_rules(self) -> None:
        from chess_variant_sandbox.loop.archive import Entry
        from chess_variant_sandbox.loop.orchestrator import _FOCUS, _focus_for

        assert "elegance" not in _FOCUS
        assert not any("rules back" in advice for advice in _FOCUS.values())

        # A `v10` score still carries the family. Weakest there, it is passed over for
        # the weakest family that has something to say.
        entry = Entry(hash="h", name="g", generation=0, parent=None, status="evaluated",
                      detail={"families": {"elegance": 0.05, "depth": 0.4,
                                           "variety": 0.3, "balance": 0.9}})
        assert _focus_for(entry, None) == _FOCUS["variety"]

        only = Entry(hash="h", name="g", generation=0, parent=None, status="evaluated",
                     detail={"families": {"elegance": 0.05}})
        assert _focus_for(only, None) == ""
