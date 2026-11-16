"""A visitor's browser score carries the dials a library score carries (R-S6e).

`docs/reference/v14-registration.md` §15. `v14`'s `decision_cost` reads the probe's move
values and `piece_participation` the pieces followed, both written by the analysis pass. A
library run's pass writes them (`analyse_corpus`, by default since R-S6d); until R-S6e the
browser's sliced pass did not, and the server's rebuild of a browser's analysis
(`site/legality.py`) kept only `near_best`, so a visitor's `v14` score had both dials absent
under the same label. The claims here:

* **a visitor's run, walked and scored by the server, has both dials** (`decision_cost` on
  the one-run rule, `piece_participation`), and they are the library pass's own readings of
  the same games (`TestAVisitorsRunHasTheDials`);
* **a browser score and a library score of one weights version carry the same dials, or
  their stamps say they do not** (`TestTheStampSaysWhichDials`): the guard that fails if the
  two ever part again without `Basis.readings` naming it.

A score stored before this, under the old basis, reads stale by the existing rule:
`test_site_provisional.TestItShowsOnlyUnderAMatchingBasis` holds it through the store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from test_site_legality import Measured, measure_as_a_browser

from chess_variant_sandbox import corpus, measurement
from chess_variant_sandbox.measurement import Basis
from chess_variant_sandbox.site import legality
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.tournament import slices
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, analyse_corpus

REPOSITORY = Path(__file__).resolve().parents[1]

# The two dials that read the pass's `v14` readings (`metrics/components.py`).
DIALS = ("decision_cost", "piece_participation")

# Dobutsu drops pieces, so a piece entering later is followed too; two searching players,
# twenty-six games a pairing: enough sampled positions for `decision_cost`'s one-run fit
# (`decision_cost.MIN_POSITIONS` a half) in a couple of seconds.
ROSTER = slices.canonical_config(
    agents=("greedy", "minimax:nodes=100"), games_per_pairing=26, seed=0, workers=1
)


@dataclass
class Both:
    """One visitor's run, scored by the server, and the library pass over the same games."""

    browser: Measured
    server: dict[str, Any]
    server_basis: Basis
    library: dict[str, Any]
    library_basis: Basis
    library_analysis: CorpusAnalysis


def _index(files: dict[str, str]) -> dict[str, Any]:
    found: dict[str, Any] = json.loads(
        next(text for name, text in files.items() if name.startswith("index-"))
    )
    return found


@pytest.fixture(scope="module")
def both(tmp_path_factory: pytest.TempPathFactory) -> Both:
    spec = load_spec(REPOSITORY / "rulesets" / "dobutsu.json")
    browser = measure_as_a_browser(spec, ROSTER, tmp_path_factory.mktemp("visitor"))
    checked = browser.check(browser.upload())
    scored = legality.score(checked, browser.current.weights_version)

    records = checked.records
    analysis = analyse_corpus(spec, records, seed=ROSTER.seed, workers=1)
    out = tmp_path_factory.mktemp("library")
    gathered = slices.gather_measurement(spec, out, records, analysis, config=ROSTER)
    library_files = {path.name: path.read_text(encoding="utf-8") for path in out.iterdir()}
    return Both(
        browser=browser,
        server=_index(scored.files),
        server_basis=checked.basis or browser.current,
        library=_index(library_files),
        library_basis=corpus.current_basis(
            spec_hash(spec), weights=gathered.result.weights_version, spec=spec,
            seeds=corpus.LIBRARY_SEEDS,
        ),
        library_analysis=analysis,
    )


def carried(index: dict[str, Any]) -> set[str]:
    """Which of the two dials a score carries."""
    return {dial for dial in DIALS if dial in index["scores"]}


class TestAVisitorsRunHasTheDials:
    def test_the_servers_score_of_a_visitors_run_has_both(self, both: Both) -> None:
        assert carried(both.server) == set(DIALS)
        cost = both.server["diagnostics"]["decision_cost"]
        # A visitor's run is one run: `k` fitted on its even games reads its odd ones, and
        # the reverse (§2.1's note on change 5), never a cross-run scale it does not have.
        assert cost["rule"] == "one run: even and odd games"
        assert "withheld" not in cost
        assert cost["positions"] >= 100

    def test_they_are_the_library_passs_own_readings_of_the_same_games(
        self, both: Both
    ) -> None:
        for dial in DIALS:
            assert both.server["raw"][dial] == both.library["raw"][dial], dial
            assert both.server["scores"][dial] == both.library["scores"][dial], dial

    def test_the_stored_analysis_is_the_library_passs_to_the_byte(self, both: Both) -> None:
        stored = json.loads(both.browser.files["analysis.json"])
        assert stored == json.loads(json.dumps(both.library_analysis.as_dict()))

    def test_both_stamps_name_the_readings(self, both: Both) -> None:
        assert both.server_basis.readings == measurement.READINGS
        assert both.library_basis.readings == measurement.READINGS
        assert both.server_basis.weights_version == both.library_basis.weights_version


def _says_which(
    browser: dict[str, Any], browser_basis: Basis, library: dict[str, Any], library_basis: Basis
) -> bool:
    """The guard: one weights version, the same dials, or stamps that differ in readings."""
    if browser_basis.weights_version != library_basis.weights_version:
        return True
    return carried(browser) == carried(library) or (
        browser_basis.readings != library_basis.readings
    )


class TestTheReadingsInTheStamp:
    """`Basis.readings` (R-S6e): named by every basis taken now, and compared."""

    def now(self) -> Basis:
        spec = load_spec(REPOSITORY / "rulesets" / "dobutsu.json")
        return corpus.current_basis(spec_hash(spec), spec=spec, seeds=corpus.ONE_SEED)

    def test_a_browser_stamp_without_them_owes_a_re_analysis(self) -> None:
        now = self.now()
        assert now.readings == measurement.READINGS
        for checked in ("", measurement.LEGAL):
            found = replace(now, readings="").in_browser(checked).stale_against(
                now.in_browser(checked)
            )
            assert found.needs == "reanalyse"
            assert any("move values" in reason for reason in found.reasons)

    def test_a_stamp_naming_other_readings_owes_one_too(self) -> None:
        now = self.now()
        for stamp in (replace(now, readings="move-values"), replace(now, readings="x")):
            assert stamp.stale_against(now).needs == "reanalyse"
            assert stamp.in_browser().stale_against(now.in_browser()).needs == "reanalyse"

    def test_a_server_stamp_without_them_is_read_by_its_analysis(self) -> None:
        """The server's pass wrote both at today's `analysis` fingerprint (by default since
        R-S6d moved it), so its absence there stales nothing; with the pass moved, the
        pass's own reason says so, once."""
        now = self.now()
        assert replace(now, readings="").stale_against(now).current
        moved = replace(now, readings="", analysis="0" * 16).stale_against(now)
        assert moved.needs == "reanalyse"
        assert not any("move values" in reason for reason in moved.reasons)

    def test_it_round_trips_and_is_written_only_where_set(self) -> None:
        now = self.now()
        assert Basis.from_dict(json.loads(json.dumps(now.as_dict()))) == now
        assert "readings" not in replace(now, readings="").as_dict()


class TestTheStampSaysWhichDials:
    def test_a_browser_and_a_library_score_carry_the_same_dials_or_say_so(
        self, both: Both
    ) -> None:
        """Fails if a browser score and a library score of one weights version differ in
        which of `v14`'s dials they carry and the stamps do not say so."""
        assert _says_which(both.server, both.server_basis, both.library, both.library_basis)

    def test_the_guard_bites_on_the_fault_r_s6e_fixed(self, both: Both, tmp_path: Path) -> None:
        """The browser's score as it was: its analysis without the readings. Its dials
        differ from the library's; stamped as now, the guard fails; stamped as the stamp
        of a pass without the readings, it holds, because the stamp says so."""
        bare = CorpusAnalysis(
            games=[
                replace(
                    game, sample_values=None, sample_movers=None,
                    fielded_instances=None, fielded_moved=None,
                )
                for game in both.library_analysis.games
            ],
            blunders=both.library_analysis.blunders,
        )
        spec = both.browser.spec
        records = slices.parse_records(both.browser.phases[slices.TOURNAMENT])
        old = slices.gather_measurement(spec, tmp_path, records, bare, config=ROSTER).result
        old_index = {"scores": dict(old.scores)}
        assert carried(old_index) == set()
        assert not _says_which(old_index, both.server_basis, both.library, both.library_basis)
        unnamed = replace(both.server_basis, readings="")
        assert _says_which(old_index, unnamed, both.library, both.library_basis)
