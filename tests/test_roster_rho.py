"""Each component's roster dependence, carried by the weights file.

`docs/archive/plan-variety.md` §5.4, box A3 (Q52). An optional per-component field,
`roster_rho`: the rank correlation between two rosters' readings of a component's raw
measurement, against `docs/reference/roster-confound.md` §8's threshold of 0.50 -- the same
threshold `v10`'s own notes cite for dropping `gradient_persistence` and `draw_slope`.
The field is shown beside each component on the game page and in the glossary, and
nothing is drawn where it is absent.

`v10.json` is published and carries the field on no component, because it did not exist
when that file was written -- and it is not edited here to prove that: a made-up version,
copied from `v10` with the field added to two of its components, does the carrying
instead. `v11` (C2, Q391) is the first shipped version to carry it, on every component, and
the last test below holds every version from `v11` on to that.
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
from chess_variant_sandbox.metrics.glossary import explain
from chess_variant_sandbox.web import library

VERSION = "vrho"
# Two components of `v10`'s thirteen carry the field in the made-up version; the rest do
# not, so a test that only the two show it also proves the other eleven show nothing.
WITH_RHO = {"stall_rate": 0.8738, "mean_branching": 0.41}


@pytest.fixture
def roster_rho_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """`v10`, with `roster_rho` added to two components and left off the rest.

    Written beside copies of every shipped version, so `v10` itself still loads from the
    same (now redirected) directory and the two can be read side by side -- the same
    trick `tests/test_score_parts.py::six_families` uses for A2.
    """
    folder = tmp_path / "weights"
    folder.mkdir()
    for path in scoring.WEIGHTS_DIR.glob("*.json"):
        shutil.copy(path, folder / path.name)
    data = json.loads((scoring.WEIGHTS_DIR / "v10.json").read_text(encoding="utf-8"))
    data["version"] = VERSION
    for name, value in WITH_RHO.items():
        data["components"][name]["roster_rho"] = value
    (folder / f"{VERSION}.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(scoring, "WEIGHTS_DIR", folder)
    return VERSION


class TestTheGlossaryCarriesRosterRho:
    def test_a_component_with_the_field_shows_it(self, roster_rho_version: str) -> None:
        guide = explain(roster_rho_version)
        for name, value in WITH_RHO.items():
            assert guide["components"][name]["roster_rho"] == pytest.approx(value)

    def test_a_component_without_the_field_carries_no_key(
        self, roster_rho_version: str
    ) -> None:
        guide = explain(roster_rho_version)
        untouched = set(guide["components"]) - set(WITH_RHO)
        assert untouched  # sanity: v10 has more than two components
        for name in untouched:
            assert "roster_rho" not in guide["components"][name]

    def test_v10_carries_the_field_on_no_component(self) -> None:
        guide = explain("v10")
        assert guide["components"]
        for entry in guide["components"].values():
            assert "roster_rho" not in entry

    def test_a_non_numeric_or_missing_field_is_read_as_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed input never raises here -- `load_weights` above already validated
        the file enough to build a `Weights` from it; a display extra just leaves the
        field off rather than crashing the page."""
        folder = tmp_path / "weights"
        folder.mkdir()
        for path in scoring.WEIGHTS_DIR.glob("*.json"):
            shutil.copy(path, folder / path.name)
        data = json.loads((scoring.WEIGHTS_DIR / "v10.json").read_text(encoding="utf-8"))
        data["version"] = "vbad"
        data["components"]["stall_rate"]["roster_rho"] = "not a number"
        (folder / "vbad.json").write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(scoring, "WEIGHTS_DIR", folder)

        guide = explain("vbad")
        assert "roster_rho" not in guide["components"]["stall_rate"]


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


def _drawn(score: dict[str, Any], guide: dict[str, Any] | None) -> str:
    """What `scoreHtml` draws for one (score, guide), from one run of the page in node."""
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
    shown = f"scoreHtml({json.dumps(score)}, {json.dumps(guide)}, null)"
    done = node_session.run(stub + script.group(1) + "\nconsole.log(" + shown + ");")
    assert done.returncode == 0, done.stderr
    return done.stdout


class TestTheGamePageDrawsRosterRhoBesideTheComponent:
    def test_the_two_components_that_carry_it_draw_it_and_nothing_else_does(
        self, roster_rho_version: str
    ) -> None:
        guide = library.score_guide({"weights_version": roster_rho_version})
        assert guide is not None
        html = _drawn(_score(roster_rho_version), guide)
        for value in WITH_RHO.values():
            assert f"&rho; {value:.2f}" in html
        # Exactly two badges: one per component that carries the field, none for the
        # other eleven -- "nothing is drawn where it is absent" as a count, not a guess.
        assert html.count("&rho;") == len(WITH_RHO)

    def test_a_v10_score_draws_no_roster_rho_at_all(self) -> None:
        guide = library.score_guide({"weights_version": "v10"})
        html = _drawn(_score("v10"), guide)
        assert "&rho;" not in html


def _versions_from(minimum: int) -> list[tuple[str, dict[str, Any]]]:
    """Every shipped weights file named `v<N>.json` with `N >= minimum`, parsed.

    Filename-keyed rather than by the `version` field inside, matching how every other
    version on disk is named -- `v1.json` through `v10.json` -- so a file that does not
    follow the convention is not silently skipped by a test that means to catch it.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(scoring.WEIGHTS_DIR.glob("v*.json")):
        match = re.fullmatch(r"v(\d+)", path.stem)
        if not match or int(match.group(1)) < minimum:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        found.append((path.stem, data))
    return found


class TestEveryComponentFromV11OnCarriesTheField:
    # A strict xfail until `v11` C2 (Q391) wrote `v11.json` with the field on all eleven
    # components, from C1's binding column (`docs/reference/v11-calibration.md`).
    def test_every_component_of_every_version_from_v11_on_carries_roster_rho(self) -> None:
        versions = _versions_from(11)
        assert versions, "no weights version from v11 on is shipped yet"
        for version, data in versions:
            missing = [
                name
                for name, config in data["components"].items()
                if "roster_rho" not in config
            ]
            assert not missing, f"{version} has no roster_rho on {missing}"
