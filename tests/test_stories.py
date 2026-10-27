"""The landing's stories as data (`web/stories.py`; `docs/archive/plan-landing.md` §2.1, §2.2, box LP1).

What is held here: the record types no measured number and declares each pair's change
exactly; every game a story plays is replayed through `Engine` to the ending its sentences
claim, and every frame's event is the engine's; a library story's numbers are its games' own
measurement, read with their basis; a constructed story says it is an example and carries
no score; and a story whose pair stops reading as told falls out of the pool.

Most of it needs no measurement: `read_score` is replaced by scores made here. The tests that
hold a real story's numbers to its games' own `canonical.json` read a scratch copy of those
six directories (this checkout's `runs/`, else the main checkout's), and skip where neither
holds them.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_PORTAL, decode_move
from chess_variant_sandbox.measurement import main_runs_dir
from chess_variant_sandbox.spec.canonical import MECHANIC_KEYS, load_spec, spec_hash
from chess_variant_sandbox.web import candidacy, designing, library, stories

ROOT = Path(__file__).resolve().parents[1]
LIBRARY_STORIES = [s for s in stories.STORIES if s.kind == stories.LIBRARY]
# The constructed kind is tested on the examples out of the pool (ground, since LPD).
CONSTRUCTED_STORIES = [s for s in (*stories.STORIES, *stories.EXAMPLES)
                       if s.kind == stories.CONSTRUCTED]
BEATS = [(s.key, which, beat) for s in (*stories.STORIES, *stories.EXAMPLES)
         for which, beat in (("wrong", s.wrong), ("right", s.right))]


@dataclasses.dataclass
class Roots:
    runs_dir: Path
    ruleset_dir: Path = ROOT / "rulesets"
    archive_dir: Path = ROOT / "archive"


@pytest.fixture(scope="module")
def listed() -> dict[str, tuple[Path, str]]:
    return stories.library_paths(Roots(ROOT / "runs"))


def spec_of(story: stories.Story, played: stories.Played,
            listed: dict[str, tuple[Path, str]]) -> dict[str, Any]:
    if story.kind == stories.CONSTRUCTED:
        assert story.files is not None
        index = 0 if played is story.wrong.game else 1
        return load_spec(stories.HERE / story.files[index])
    return load_spec(listed[played.hash][0])


def made_score(viable: bool, status: str = "current") -> dict[str, Any]:
    """A score shaped as `library.Game.score` returns it, for the pool's tests."""
    return {"index": 0.5, "sd": 0.01, "seeds": [0, 1000, 2000, 3000], "viable": viable,
            "status": status, "revision": "v12", "measured_at": "2026-10-24",
            "certified": None, "violations": [] if viable else ["run 0: x=1 below minimum 2"],
            "verdict": {"accept": 4 if viable else 0, "refuse": 0 if viable else 4,
                        "split": ""},
            "families": {q: 0.5 for q in stories.QUALITIES}}


@pytest.fixture
def scores(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Every library game's estimate as the story says it reads: failed ruled out, fixed in.
    A test edits the dict to force a verdict."""
    made = {}
    for story in LIBRARY_STORIES:
        made[story.wrong.game.hash] = made_score(False)
        made[story.right.game.hash] = made_score(True)

    def read(app: Any, digest: str, *_: Any) -> dict[str, Any] | None:
        return copy.deepcopy(made.get(digest))

    monkeypatch.setattr(stories, "read_score", read)
    return made


# ------------------------------------------------------------------ the record


def test_four_stories_in_the_plans_order() -> None:
    """LPD (2026-10-26): *a sweep*, a measured pair, took ground's place in the pool; ground
    stays a constructed example out of it."""
    assert [s.key for s in stories.STORIES] == ["stones", "door", "blast", "sweep"]
    assert [s.kind for s in stories.STORIES] == [stories.LIBRARY] * 4
    assert [s.key for s in stories.EXAMPLES] == ["ground"]
    assert stories.GROUND.kind == stories.CONSTRUCTED


def test_the_families_differ_and_each_is_the_palettes() -> None:
    families = [s.family for s in stories.STORIES]
    assert len(set(families)) == len(families)
    assert all(family in stories.FAMILIES for family in families)
    assert len(set(stories.FAMILIES.values())) == len(stories.FAMILIES)


def _sentences(story: stories.Story) -> list[str]:
    said = [story.title, story.clause, story.chip, story.idea, story.note, story.tweak,
            story.card.title, story.card.lead, story.card.was, story.card.now,
            story.card.tail, story.quote]
    for beat in (story.wrong, story.right):
        said += [beat.say, beat.lead, beat.end, beat.source]
    return said


@pytest.mark.parametrize("story", stories.STORIES, ids=lambda s: s.key)
def test_no_number_is_typed_in_a_story(story: stories.Story) -> None:
    """Every number a story says is filled from its games or its specs: a digit in a
    sentence, outside a `{placeholder}`, would be a number nobody can hold to a game."""
    for text in _sentences(story):
        bare = re.sub(r"\{[a-z_]+\}", "", text)
        assert not re.search(r"\d", bare), text


def _leaves(a: Any, b: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Every path at which two mechanics differ, worked out without the story's change."""
    if isinstance(a, dict) and isinstance(b, dict):
        found = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                found.append((*path, key))
            else:
                found += _leaves(a[key], b[key], (*path, key))
        return found
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b, strict=True))
                for p in _leaves(x, y, (*path, i))]
    return [] if a == b else [path]


@pytest.mark.parametrize("story", stories.STORIES, ids=lambda s: s.key)
def test_each_pair_differs_by_exactly_the_declared_change(
        story: stories.Story, listed: dict[str, tuple[Path, str]]) -> None:
    failed = spec_of(story, story.wrong.game, listed)
    fixed = spec_of(story, story.right.game, listed)
    assert spec_hash(failed) == story.wrong.game.hash
    assert spec_hash(fixed) == story.right.game.hash
    assert stories.differ_by_change(failed, fixed, story.changes)
    before, after = ({k: s[k] for k in MECHANIC_KEYS if k in s} for s in (failed, fixed))
    assert sorted(map(str, _leaves(before, after))) == sorted(str(c.path) for c in story.changes)


def test_a_change_that_is_not_the_pairs_does_not_hold(
        listed: dict[str, tuple[Path, str]]) -> None:
    story = stories.BY_KEY["stones"]
    failed = spec_of(story, story.wrong.game, listed)
    fixed = spec_of(story, story.right.game, listed)
    wrong = (stories.Change(("rules", "drops", "stock", "X"), "unlimited", 4),)
    assert not stories.differ_by_change(failed, fixed, wrong)
    stale = (stories.Change(("rules", "drops", "stock", "X"), 2, 3),)
    assert not stories.differ_by_change(failed, fixed, stale)


def test_the_change_is_said_as_the_rule_language_reads_it() -> None:
    lines = stories.change_lines(stories.BY_KEY["stones"].changes)
    assert lines == [{"del": '"X": "unlimited"', "ins": '"X": 3'}]
    lines = stories.change_lines(stories.BY_KEY["blast"].changes)
    assert lines == [{"del": '"of": ["Q"]', "ins": ""}, {"del": "", "ins": '"by": ["Q"]'}]


def test_library_games_are_listed_and_constructed_ones_are_not(
        listed: dict[str, tuple[Path, str]]) -> None:
    for story in LIBRARY_STORIES:
        for played in (story.wrong.game, story.right.game):
            assert played.hash in listed, story.key
            assert listed[played.hash][1] in ("committed", "archived")
    shelves = {spec_hash(load_spec(p)) for d in ("rulesets", "archive")
               for p in (ROOT / d).glob("*.json")}
    for story in CONSTRUCTED_STORIES:
        for played in (story.wrong.game, story.right.game):
            assert played.hash not in listed and played.hash not in shelves
            assert played.example is not None
        assert story.files is not None
        for file in story.files:
            spec = load_spec(stories.HERE / file)
            assert "example" in str(spec.get("description", "")).lower()


def test_the_quoted_words_are_the_fixed_games_own(listed: dict[str, tuple[Path, str]]) -> None:
    for story in LIBRARY_STORIES:
        fixed = spec_of(story, story.right.game, listed)
        assert story.quote and story.quote in fixed["description"], story.key


# ------------------------------------------------------------------ the games


@pytest.fixture(scope="module")
def replays(listed: dict[str, tuple[Path, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for key, which, beat in BEATS:
        story = stories.BY_KEY[key]
        out[(key, which)] = stories.replay(spec_of(story, beat.game, listed), beat.game)
    return out


@pytest.mark.parametrize(("key", "which", "beat"), BEATS,
                         ids=[f"{k}-{w}" for k, w, _ in BEATS])
def test_each_recorded_game_ends_as_its_sentence_says(
        key: str, which: str, beat: stories.Beat,
        replays: dict[tuple[str, str], dict[str, Any]]) -> None:
    game = replays[(key, which)]
    winner, reason = beat.game.ending
    assert game["outcome"] == {"winner": winner, "reason": reason}
    record = stories.game_record(beat.game)
    assert game["plies"] == record["plies"] == len(record["moves"])
    frames = game["frames"]
    assert frames[-1]["n"] == game["plies"] and frames[-1]["key"]
    assert any(f["n"] == game["moment"] and f["key"] for f in frames)


def _board_after(game: dict[str, Any]) -> list[int]:
    board = list(game["start"]["board"])
    for frame in game["frames"]:
        for square, value in frame["set"]:
            board[square] = value
    return board


def test_what_each_ending_shows(replays: dict[tuple[str, str], dict[str, Any]]) -> None:
    """The end sentences' claims, on the engine's last positions."""
    # Stones: every square filled, and the player to move has no move.
    assert 0 not in _board_after(replays[("stones", "wrong")])
    # Three stones: six stones dropped and both hands empty at the end.
    three = replays[("stones", "right")]["frames"]
    assert sum(1 for f in three if "drop" in f) == 6
    assert three[-1]["hands"] == [[0] * len(three[-1]["hands"][0])] * 2
    # A door: through it on the first move, both in and out ends named.
    first = replays[("door", "wrong")]["frames"][0]
    assert first["n"] == 1 and len(first["portal"]) == 2
    assert "portal" in replays[("door", "right")]["frames"][27]
    # A blast: the moment's blast clears a royal piece and so ends the game.
    blast = replays[("blast", "wrong")]["frames"][-1]["blast"]
    assert blast["at"] in blast["area"]
    bomb = replays[("blast", "right")]
    assert [f["n"] for f in bomb["frames"]] == [*range(1, 14), 63]
    assert bomb["frames"][-1]["cut"] and "blast" in bomb["frames"][12]
    # A sweep: mate on the first move, the sweep clearing a square on the way; the fix's
    # moment sweeps two in one move.
    sweep = replays[("sweep", "wrong")]["frames"]
    assert len(sweep) == 1 and sweep[0]["sweep"] and sweep[0]["key"]
    assert len(replays[("sweep", "right")]["frames"][23]["sweep"]) == 2
    # Ground: repainted, the count never reaches the goal; kept for good, it does.
    painted = replays[("ground", "wrong")]
    goal = 25
    assert painted["facts"]["best"] < goal
    assert replays[("ground", "right")]["frames"][-1]["counts"][0] == goal


@pytest.mark.parametrize(("key", "which", "beat"), BEATS,
                         ids=[f"{k}-{w}" for k, w, _ in BEATS])
def test_every_frames_event_is_the_engines(
        key: str, which: str, beat: stories.Beat,
        replays: dict[tuple[str, str], dict[str, Any]]) -> None:
    game = replays[(key, which)]
    moves = stories.game_record(beat.game)["moves"]
    for frame in game["frames"]:
        origin, target, kind, _ = decode_move(int(moves[frame["n"] - 1]))
        assert frame["to"] == target
        if "portal" in frame:
            assert kind == MOVE_PORTAL
        if "drop" in frame:
            assert kind == MOVE_DROP and frame["drop"] == target and "from" not in frame
        else:
            assert frame["from"] == origin
        if "blast" in frame:
            area = set(frame["blast"]["area"])
            assert set(frame["blast"]["cleared"]) <= area
            assert frame["blast"]["cleared"]
        if "sweep" in frame:
            assert frame["sweep"] and target not in frame["sweep"] and origin not in frame["sweep"]
        if "counts" in frame:
            assert all(owner in (0, 1, 2) for _, owner in frame["claims"])


def test_a_game_the_engine_does_not_play_so_is_refused(
        listed: dict[str, tuple[Path, str]], monkeypatch: pytest.MonkeyPatch) -> None:
    story = stories.BY_KEY["door"]
    record = stories.game_record(story.wrong.game)
    record["moves"] = [record["moves"][1], *record["moves"][1:]]
    monkeypatch.setattr(stories, "game_record", lambda played: record)
    with pytest.raises(stories.StoryError):
        stories.replay(spec_of(story, story.wrong.game, listed), story.wrong.game)


# ------------------------------------------------------------------ rendered


@pytest.mark.parametrize("story", CONSTRUCTED_STORIES, ids=lambda s: s.key)
def test_a_constructed_story_says_example_and_carries_no_score(
        story: stories.Story, scores: dict[str, Any], tmp_path: Path) -> None:
    told = stories.render_story(story, Roots(tmp_path))
    said = stories.text(told)
    assert told["example"] and "example" in said
    assert "runs" not in said and "±" not in said
    assert not re.search(r"\b0\.\d", said)
    for side in (told["failed"], told["fixed"]):
        assert side["line"] == stories.EXAMPLE_LINE
        assert not {"index", "sd", "runs", "qualities", "revision", "certified"} & set(side)


@pytest.mark.parametrize("story", CONSTRUCTED_STORIES, ids=lambda s: s.key)
def test_a_constructed_storys_quick_answers_are_the_designers(
        story: stories.Story, scores: dict[str, Any], tmp_path: Path) -> None:
    told = stories.render_story(story, Roots(tmp_path))
    assert story.files is not None
    for side, file in zip((told["failed"], told["fixed"]), story.files, strict=True):
        raw = json.loads((stories.HERE / file).read_text(encoding="utf-8"))
        checked = designing.check(raw)
        screened = candidacy.screen_draft(raw)
        quick = side["quick"]
        assert quick["playable"] == checked["playable"]
        assert quick["opening"] == checked["openingWords"]
        assert quick["screen"]["passed"] == screened["passed"]
        assert quick["screen"]["decisive"] == int(screened["stats"]["decisive"])


def test_every_story_holds_while_its_pair_reads_as_told(scores: dict[str, Any],
                                                         tmp_path: Path) -> None:
    rendered = stories.render(Roots(tmp_path))
    assert rendered["pool"] == ["stones", "door", "blast", "sweep"], rendered["dropped"]
    assert rendered["dropped"] == []


@pytest.mark.parametrize("force", ["failed turns viable", "fixed ruled out",
                                   "fixed provisional", "failed unmeasured"])
def test_a_story_whose_pair_stops_reading_as_told_leaves_the_pool(
        force: str, scores: dict[str, Any], tmp_path: Path) -> None:
    story = stories.BY_KEY["stones"]
    failed, fixed = story.wrong.game.hash, story.right.game.hash
    if force == "failed turns viable":
        scores[failed] = made_score(True)
    elif force == "fixed ruled out":
        scores[fixed] = made_score(False)
    elif force == "fixed provisional":
        scores[fixed] = made_score(True, status="provisional")
    else:
        del scores[failed]
    rendered = stories.render(Roots(tmp_path))
    assert "stones" not in rendered["pool"]
    assert "stones" not in [s["key"] for s in rendered["stories"]]
    assert [d["key"] for d in rendered["dropped"]] == ["stones"]
    assert rendered["pool"] == ["door", "blast", "sweep"]


def test_with_every_library_pair_gone_a_constructed_story_stays(
        scores: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for digest in list(scores):
        scores[digest] = made_score(True, status="provisional")
    assert stories.render(Roots(tmp_path))["pool"] == []
    monkeypatch.setattr(stories, "STORIES", (*stories.STORIES, stories.GROUND))
    assert stories.render(Roots(tmp_path))["pool"] == ["ground"]


def test_a_constructed_story_that_stops_replaying_leaves_the_pool(
        scores: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = stories.BY_KEY["ground"]
    broken = dataclasses.replace(
        story, right=dataclasses.replace(
            story.right, game=dataclasses.replace(story.right.game, ending=(1, "coloured"))))
    monkeypatch.setattr(stories, "STORIES", (*stories.STORIES, broken))
    rendered = stories.render(Roots(tmp_path))
    assert rendered["pool"] == ["stones", "door", "blast", "sweep"]
    assert [d["key"] for d in rendered["dropped"]] == ["ground"]


def test_the_sentences_are_filled_from_the_games(scores: dict[str, Any],
                                                 tmp_path: Path) -> None:
    told = {s["key"]: s for s in stories.render(Roots(tmp_path))["stories"]}
    for story in told.values():
        for beat in ("wrong", "right"):
            for part in ("say", "lead", "end", "source"):
                assert "{" not in story[beat][part]
    assert told["door"]["right"]["end"].startswith("Turn 28: Black's rook goes in on b3")
    assert told["blast"]["wrong"]["lead"] == "The hardest computer player against a weak one. Watch d8."
    assert told["stones"]["card"]["before"] == stories.rule_line(
        load_spec(ROOT / "archive" / "stones.json"), ("rules", "drops"))


# ------------------------------------------------------------------ against runs/


def _runs_with_the_stories() -> Path | None:
    first = LIBRARY_STORIES[0].wrong.game
    for runs in (ROOT / "runs", main_runs_dir()):
        if runs is not None and (runs / first.hash / "canonical.json").exists():
            return runs
    return None


@pytest.fixture(scope="module")
def measured_runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scratch copy of the six games' `runs/<hash>/` (dates kept), so reading them can
    never write into a checkout's `runs/`."""
    source = _runs_with_the_stories()
    if source is None:
        pytest.skip("no runs/ here holds the stories' measurements")
    copied = tmp_path_factory.mktemp("story-runs")
    for story in LIBRARY_STORIES:
        for played in (story.wrong.game, story.right.game):
            shutil.copytree(source / played.hash, copied / played.hash,
                            ignore=shutil.ignore_patterns("previous"))
    return copied


def test_a_library_storys_numbers_are_its_games_own_measurement(measured_runs: Path) -> None:
    rendered = stories.render(Roots(measured_runs))
    told = {s["key"]: s for s in rendered["stories"]}
    for story in LIBRARY_STORIES:
        stored = {played.hash: json.loads(
            (measured_runs / played.hash / "canonical.json").read_text(encoding="utf-8"))
            for played in (story.wrong.game, story.right.game)}
        reads_as_told = (not stored[story.wrong.game.hash]["viable"]
                         and stored[story.right.game.hash]["viable"])
        if not reads_as_told:
            assert story.key not in rendered["pool"]
            continue
        if story.key not in told:  # stale here: dropped, never shown provisional
            assert story.key in [d["key"] for d in rendered["dropped"]]
            continue
        for side, played in (("failed", story.wrong.game), ("fixed", story.right.game)):
            said, canonical = told[story.key][side], stored[played.hash]
            assert said["index"] == canonical["index"]
            assert said["runs"] == len(canonical["seeds"]) and said["runs"] >= 1
            assert said["viable"] == canonical["viable"]
            assert said["status"] == "current"
            assert f"{canonical['index']:.3f}" in said["line"]
            assert f"{said['runs']} run" in said["line"]
            assert "scoring rules version" in said["line"]
            assert set(said["qualities"]) == set(stories.QUALITIES)
        assert told[story.key]["failed"]["words"]


def test_the_recorded_moves_are_the_runs_own_records() -> None:
    source = _runs_with_the_stories()
    if source is None:
        pytest.skip("no runs/ here holds the stories' recorded games")
    for story in LIBRARY_STORIES:
        for played in (story.wrong.game, story.right.game):
            path = source / f"{played.hash}-{played.seed}" / "games.jsonl"
            if not path.exists():
                pytest.skip(f"{path} is not here")
            assert played.line is not None
            with path.open(encoding="utf-8") as rows:
                row = json.loads(next(r for i, r in enumerate(rows) if i == played.line))
            kept = stories.game_record(played)
            assert row["ruleset_hash"] == played.hash
            assert kept["moves"] == row["moves"]
            assert (kept["winner"], kept["reason"]) == (row["winner"], row["reason"])
            assert kept["agents"] == row["agents"]


def test_render_is_read_only_and_needs_no_listing(scores: dict[str, Any], tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """`render` never lists the library (a cold listing is seconds) and writes nothing."""
    monkeypatch.setattr(library, "discover", lambda *a, **k: pytest.fail("listed"))
    stories.render(Roots(tmp_path))
    assert list(tmp_path.iterdir()) == []
