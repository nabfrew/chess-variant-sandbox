"""The rebuilt retired specs are the games that left the library (Q868)."""

from __future__ import annotations

import pytest
from retired_specs import REBUILT, RETIRED_HASHES, RETIRED_PREVIEWS, retired_preview

from chess_variant_sandbox.spec import spec_hash, validate_spec


@pytest.mark.parametrize("name", sorted(REBUILT))
def test_each_rebuild_is_the_retired_game(name: str) -> None:
    spec = REBUILT[name]()
    assert validate_spec(spec) == []
    assert spec_hash(spec)[:12] == RETIRED_HASHES[name]


@pytest.mark.parametrize("name", sorted(REBUILT))
def test_none_of_them_is_back_in_the_library(name: str) -> None:
    from retired_specs import ARCHIVE

    assert not (ARCHIVE / f"{name}.json").exists()
    assert not (ARCHIVE.parent / "rulesets" / f"{name}.json").exists()


@pytest.mark.parametrize("name", sorted(RETIRED_PREVIEWS))
def test_each_retired_preview_is_kept_whole_and_out_of_the_library(name: str) -> None:
    """Q962b's six: the files moved unchanged, so each is valid and keeps its hash, and none is
    back in `archive/` or `rulesets/`, where the site would list it again."""
    from retired_specs import ARCHIVE

    spec = retired_preview(name)
    assert validate_spec(spec) == []
    assert spec_hash(spec)[:12] == RETIRED_PREVIEWS[name]
    assert not (ARCHIVE / f"{name}.json").exists()
    assert not (ARCHIVE.parent / "rulesets" / f"{name}.json").exists()
