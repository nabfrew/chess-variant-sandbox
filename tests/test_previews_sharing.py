"""Q1080: a game's page names the previews that share a rule of its beyond the reference game.

The user, 2026-10-31: *"The changelog names four unnamed pushing games being added, I can't
find them."* A preview is in no near list, so a pushing game's page never named the push
previews. `library.previews_sharing` reads the specs alone: a preview is named on a game's
page when the two share a rule (`relate.rules_involved`) the reference game does not have.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chess_variant_sandbox.web import library

ROOT = Path(__file__).resolve().parent.parent
PUSH_PREVIEWS = {
    "los-alamos-with-a-closed-edge.json": "daa6917bd359",
    "los-alamos-with-shove-or-take.json": "1fced818630e",
    "los-alamos-with-crushing-pillars.json": "e72a8715217c",
    "gardner-with-bishops-that-shove-straight.json": "e271933cd127",
}
LIBRARY_FILES = ("chess.json", "brink.json")


@pytest.fixture(scope="module")
def games(tmp_path_factory: pytest.TempPathFactory) -> dict[str, library.Game]:
    root = tmp_path_factory.mktemp("sharing")
    for folder in ("rulesets", "archive", "runs"):
        (root / folder).mkdir()
    for name in LIBRARY_FILES:
        shutil.copy(ROOT / "rulesets" / name, root / "rulesets" / name)
    for name in PUSH_PREVIEWS:
        shutil.copy(ROOT / "archive" / name, root / "archive" / name)
    found = library.discover(root / "rulesets", root / "runs", root / "archive")
    return {game.path.name: game for game in found}


def _named(games: dict[str, library.Game], name: str) -> list[str]:
    return [one["hash"] for one in library.previews_sharing(games[name], games.values())]


def test_a_pushing_game_names_every_push_preview(games: dict[str, library.Game]) -> None:
    assert sorted(_named(games, "brink.json")) == sorted(PUSH_PREVIEWS.values())


def test_the_reference_game_names_none(games: dict[str, library.Game]) -> None:
    """Nothing is beyond the reference game itself, so its page names no preview."""
    assert _named(games, "chess.json") == []


def test_a_preview_names_the_others_and_never_itself(games: dict[str, library.Game]) -> None:
    for name, digest in PUSH_PREVIEWS.items():
        named = _named(games, name)
        assert digest not in named
        assert sorted(named) == sorted(set(PUSH_PREVIEWS.values()) - {digest})


def test_without_the_reference_game_nothing_is_named(games: dict[str, library.Game]) -> None:
    others = [game for name, game in games.items() if name != "chess.json"]
    assert library.previews_sharing(games["brink.json"], others) == []


def test_the_game_page_draws_them_as_links() -> None:
    from page_text import PAGE

    assert "previewsSharingHtml(game)" in PAGE
    assert "game.previewsSharing" in PAGE
