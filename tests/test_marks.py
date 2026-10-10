"""R2 of `docs/plan-return.md` §8 (§6.3): a member's generated mark.

An invader piece of their own, in the family of the site's invader rook, generated from
the account's id alone: nothing stored, nothing uploaded, the handle never read. These
hold the rule; where it shows (the header's corner, your page, the players, a match's
seats, *Play a person*) is held by the payload tests beside each route and the markup
checks below.
"""

from __future__ import annotations

import inspect
import re
from collections import Counter

from page_text import PAGE
from test_style import _delta_e, _luminance, _rgb, _theme

from chess_variant_sandbox.site import identity, pages
from chess_variant_sandbox.web.page import marks, shell, tokens

IDS = range(1, 2001)


class TestTheRule:
    def test_a_figure_is_sixteen_by_sixteen_and_mirrored_like_the_rook(self) -> None:
        for user in (1, 2, 3, 77, 10_000, 2**40):
            rows = marks.rows(user)
            assert len(rows) == 16
            for row in rows:
                assert len(row) == 16 and set(row) <= {"#", "."}
                assert row == row[::-1]

    def test_it_is_a_function_of_the_id_alone(self) -> None:
        assert list(inspect.signature(marks.mark).parameters) == ["user"]
        for user in IDS[:50]:
            assert marks.mark(user) == marks.mark(user)
        # Nothing else feeds it: no handle, no clock, no store.
        source = inspect.getsource(marks)
        assert "random" not in source and "time" not in source.replace("times", "")

    def test_people_get_different_marks(self) -> None:
        figures = Counter(marks.rows(user) for user in IDS)
        drawn = Counter((marks.rows(user), marks.hue(user)) for user in IDS)
        assert len(figures) > 1000
        # Among the first thirty people, nobody shares a mark.
        assert len({(marks.rows(user), marks.hue(user)) for user in range(1, 31)}) == 30
        assert max(drawn.values()) <= 4

    def test_eight_who_join_one_after_another_wear_eight_colours(self) -> None:
        for first in (1, 2, 9, 1000):
            assert {marks.hue(user) for user in range(first, first + marks.HUES)} == set(
                range(1, marks.HUES + 1)
            )

    def test_every_part_and_every_hue_is_reached(self) -> None:
        crests = Counter(marks.rows(user)[:3] for user in IDS)
        bases = Counter(marks.rows(user)[11:] for user in IDS)
        hues = Counter(marks.hue(user) for user in IDS)
        assert len(crests) == len(marks.CRESTS)
        assert len(bases) == len(marks.BASES)
        assert set(hues) == set(range(1, marks.HUES + 1))

    def test_every_figure_has_two_eyes_in_a_solid_head(self) -> None:
        """The head's middle rows always hold an empty pair inside the ink, which is what
        makes a figure read as a creature at 18 px rather than a block."""
        for user in IDS[:300]:
            rows = marks.rows(user)
            eyes = [row for row in rows[5:7] if re.search(r"#\.+#", row[:8])]
            assert eyes, user

    def test_the_path_is_the_rook_s_kind_of_path(self) -> None:
        drawn = marks.mark(5)
        assert drawn["d"] == shell.mark_path(marks.rows(5))
        assert re.fullmatch(r"[Mhvz0-9 -]+", drawn["d"])
        assert 1 <= drawn["hue"] <= marks.HUES


class TestWhatIsDrawn:
    def test_the_server_s_drawing_reads_the_hue_s_token(self) -> None:
        drawn = marks.svg(5, 40)
        assert f'fill="var(--mk-{marks.hue(5)})"' in drawn
        assert 'width="40"' in drawn and 'class="memberMark"' in drawn
        assert 'shape-rendering="crispEdges"' in drawn

    def test_every_hue_is_a_token_in_both_themes(self) -> None:
        for n in range(1, marks.HUES + 1):
            assert f"mk-{n}" in tokens.LIGHT and f"mk-{n}" in tokens.DARK


class TestTheHues:
    """R3: the eight hues are the club room's, set in both themes. Pre-registered
    (Q598, before the hues were picked): each reads at 3 : 1 on the ground and the panel
    over the paper's worst fleck (`tokens.FLOORS`, run by `test_style`); each is at least
    30 CIELAB units from the action's green and the alarm's red, so a mark never reads as
    a button or a danger (S4); and any two are at least 15 apart, so two people's marks
    differ in colour as well as in shape."""

    APART_FROM_MEANINGS = 30.0
    APART_FROM_EACH_OTHER = 15.0

    def _hues(self, dark: bool) -> dict[str, tuple[float, float, float]]:
        theme = _theme(dark)
        return {f"mk-{n}": _rgb(theme[f"mk-{n}"]) for n in range(1, marks.HUES + 1)}

    def test_every_hue_has_its_floor_on_the_page_and_a_panel(self) -> None:
        for n in range(1, marks.HUES + 1):
            for back in ("panel", "bg"):
                assert (f"mk-{n}", back, 3.0) in tokens.FLOORS

    def test_clear_of_the_action_and_the_alarm(self) -> None:
        for dark in (False, True):
            theme = _theme(dark)
            for name, hue in self._hues(dark).items():
                for meaning in ("act", "alarm"):
                    apart = _delta_e(hue, _rgb(theme[meaning]))
                    assert apart >= self.APART_FROM_MEANINGS, (dark, name, meaning, round(apart))

    def test_clear_of_each_other(self) -> None:
        for dark in (False, True):
            hues = self._hues(dark)
            for one in hues:
                for two in hues:
                    if one < two:
                        apart = _delta_e(hues[one], hues[two])
                        assert apart >= self.APART_FROM_EACH_OTHER, (dark, one, two, round(apart))

    def test_dark_lightens_every_hue_and_never_repeats_light(self) -> None:
        for n in range(1, marks.HUES + 1):
            light, dark = tokens.LIGHT[f"mk-{n}"], tokens.DARK[f"mk-{n}"]
            assert light != dark
            assert _luminance(_rgb(dark)) > _luminance(_rgb(light))

    def test_the_mock_s_placeholders_are_gone(self) -> None:
        placeholders = {"#c2410c", "#0f766e", "#7c3aed", "#b45309", "#be185d", "#2563eb",
                        "#15803d", "#475569"}
        for theme in (tokens.LIGHT, tokens.DARK):
            assert not placeholders & {theme[f"mk-{n}"] for n in range(1, marks.HUES + 1)}

    def test_somebody_who_left_has_no_mark(self) -> None:
        assert identity.mark_of(5, identity.FORMER) is None
        assert identity.mark_of(None, "ada") is None and identity.mark_of(0, "ada") is None
        assert identity.mark_of(5, "ada") == marks.mark(5)

    def test_one_builder_draws_it_on_both_servers_as_elements(self) -> None:
        script = shell.SHELL_SCRIPT_FILE.read_text(encoding="utf-8")
        start = script.index("window.memberMark = function")
        builder = script[start:]
        assert "createElementNS" in builder and "innerHTML" not in builder
        # A path is only ever a path: anything else draws nothing.
        assert "/^[MhvzZ0-9 -]*$/.test(mark.d)" in builder
        assert "'var(--mk-' + (Number(mark.hue) || 1) + ')'" in builder

    def test_it_is_shown_where_a_person_is_named(self) -> None:
        # The application's header, a match's seats, Play a person, games waiting on you.
        assert "memberButton(VIEWER, VIEWER_MARK)" in PAGE
        assert "window.memberMark(markData, 20)" in PAGE
        assert "markHtml(between.marks[player], 18)" in PAGE
        assert "markHtml(person.mark, 20)" in PAGE
        assert "markHtml(entry.opponentMark, 18)" in PAGE
        assert "markHtml(entry.fromMark, 18)" in PAGE
        # The site's own pages: the players' table and your correspondence.
        players = (pages.SCRIPT_DIR / "players.js").read_text(encoding="utf-8")
        profile = (pages.SCRIPT_DIR / "profile.js").read_text(encoding="utf-8")
        assert "window.memberMark(person.mark, 20)" in players
        assert "window.memberMark(entry.opponentMark, 18)" in profile
        assert "window.memberMark(entry.fromMark, 18)" in profile

    def test_your_page_and_the_corner_carry_your_mark(self) -> None:
        page = pages.profile("ada", user=7)
        assert f"<h1>{marks.svg(7, 40)}ada</h1>" in page
        assert f'id="who">{marks.svg(7, 20)}ada</span>' in page
        assert pages._member_button("ada", marks.svg(7, 20)) in pages.players("ada", user=7)
        assert f'{marks.svg(7, 20)}<span class="memberName">ada</span>' in pages.players(
            "ada", user=7)
        # Without an id (a test, a constant page), the corner is the handle alone.
        assert 'id="who">ada</span>' in pages.profile("ada")

    def test_the_corner_drops_it_on_a_phone(self) -> None:
        assert "header.top #who .memberMark { display: none; }" in pages.STYLE
