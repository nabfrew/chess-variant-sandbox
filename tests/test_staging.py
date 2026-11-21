"""deploy/staging.py stays a copy of what the deploy does (PW2, publish-walk.md §6).

No Docker here: these read the files the staging copy is built from and hold them to the
deploy's own, so a change to the box's limits, its front or copy-runs.sh's selection that
staging does not follow fails here rather than in the next walk.
"""

from __future__ import annotations

import re

import private_paths
import pytest

READS = (
    "deploy/staging.py",
    "deploy/staging/Dockerfile",
    "deploy/copy-runs.sh",
    "deploy/remote/chess-variant-sandbox-serve.service",
    "deploy/remote/Caddyfile",
    "deploy/remote/postgres.sh",
)
pytestmark = private_paths.needs(*READS)

staging = private_paths.script("deploy/staging.py", "staging")


def _read(relative: str) -> str:
    return private_paths.path(relative).read_text(encoding="utf-8")


def test_the_runs_selection_is_copy_runs_own():
    text = _read("deploy/copy-runs.sh")
    theirs = re.findall(r"--exclude='([^']+)'", text)
    assert theirs, "copy-runs.sh's tar names no --exclude"
    assert tuple(theirs) == staging.COPY_RUNS_EXCLUDES


@pytest.mark.parametrize(
    ("path", "left_out"),
    [
        ("0123456789ab/canonical.json", False),
        ("0123456789ab/previous/1/canonical.json", True),
        ("opponents/0123456789ab.json", False),
        ("opponents/0123456789ab.games.jsonl", True),
        ("opponents/before-2026-10-01/x.json", True),
        ("puzzles/0123456789ab.json", False),
        ("puzzles/work/a.json", True),
        ("puzzles/x.tmp", True),
        ("endgames/0123456789ab.json", False),
    ],
)
def test_excluded_reads_the_patterns_as_tar_does(path, left_out):
    assert staging.excluded(path) is left_out


def test_the_limits_are_the_serve_units():
    limits = staging.unit_limits()
    unit = _read("deploy/remote/chess-variant-sandbox-serve.service")
    for key, value in limits.items():
        assert f"\n{key}={value}\n" in unit
    flags = staging.docker_limits(limits)
    memory = limits["MemoryMax"].lower()
    assert flags[flags.index("--memory") + 1] == memory
    assert flags[flags.index("--memory-swap") + 1] == memory  # the box has no swap
    assert float(flags[flags.index("--cpus") + 1]) == int(limits["CPUQuota"].rstrip("%")) / 100


def test_the_front_is_the_boxs_caddyfile_with_only_its_address_changed():
    template = _read("deploy/remote/Caddyfile")
    made = staging.caddyfile(template)
    assert "@DOMAIN@" not in made and "@ACME_EMAIL@" not in made
    assert ":80 {" in made
    site = template[template.index("\n@DOMAIN@ {") + len("\n@DOMAIN@ {"):]
    assert made.endswith(site), "every route of the box's site block is served unchanged"


def test_the_image_runs_the_boxs_own_build_scripts():
    dockerfile = _read("deploy/staging/Dockerfile")
    assert "deploy/remote/toolchain.sh" in dockerfile
    assert "deploy/remote/build.sh build" in dockerfile
    assert "deploy/remote/build.sh verify" in dockerfile
    assert "FROM ${BASE}" in dockerfile and "ubuntu:26.04" in dockerfile


def test_the_app_reads_the_boxs_site_env():
    flags = staging.app_env(18480)
    assert flags[flags.index("--env-file") + 1].endswith("chess-variant-sandbox-site.env")
    assert "MAIL_API_KEY=staging-not-a-key" in flags


def test_invitations_go_only_to_invented_addresses():
    with pytest.raises(SystemExit):
        staging.invite("someone@example.com", send=False)


def test_the_postgres_settings_are_postgres_sh_own():
    text = _read("deploy/remote/postgres.sh")
    for key, value in staging.PG_SETTINGS.items():
        if key == "listen_addresses":
            assert "listen_addresses = ''" in text
        else:
            assert f"{key} = {value}" in text, key
