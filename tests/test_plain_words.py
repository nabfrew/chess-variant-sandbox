"""W5b (`docs/archive/plan-understanding.md` §3.3 D40-D50, M7): the rules' complaints in plain words.

Three things are held here. The designer says what is wrong with a draft in the names it
draws each part by (`spec.plain`), never by a path into the game's text. The words changed
and the conditions did not: the validator's own sentences, which the proposing LLM repairs
a draft from, are the same ones, and `plain_problems` is one problem out for each problem in.
And the guard (`site_words`) now refuses the phrases this task removed, read from where
they are said: the problems, the gate's reasons, the rule cards and the small changes.
"""

from __future__ import annotations

import copy
import random
import re
from pathlib import Path
from typing import Any

import pytest
import site_words

from chess_variant_sandbox.loop import gates
from chess_variant_sandbox.loop.operators import OPERATORS
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.plain import FIELD_WORDS, KIND_WORDS, plain_problem, plain_problems
from chess_variant_sandbox.web import cards, designing

ROOT = Path(__file__).resolve().parents[1]


def _hits(text: str) -> list[str]:
    """Every phrase of the guard, `NOT_YET`'s included: what this task removed is gone
    from everything read here even while something elsewhere still serves it."""
    return site_words.hits(text, tuple(site_words.PHRASES))
# Three shapes of game: a square board with every ordinary rule, a hexagon, and one with
# wild pieces, walls and drops, so the problems reach the rule cards' own fields.
DRAFTS = ("chess", "glinski-hexagonal", "crazyhouse")


def _spec(name: str) -> dict[str, Any]:
    path = ROOT / "rulesets" / f"{name}.json"
    if not path.exists():
        path = next((ROOT / "rulesets").glob("*.json"))
    return designing.settled(load_spec(path))


def _broken(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The same draft broken twenty ways, each a mistake a person can make in the form."""
    out: dict[str, dict[str, Any]] = {}

    def made(name: str, change: Any) -> None:
        draft = copy.deepcopy(spec)
        change(draft)
        out[name] = draft

    made("no name", lambda s: s.pop("name"))
    made("blank name", lambda s: s.update(name=" "))
    made("z9", lambda s: s["setup"]["placements"][0].update(squares=["z9"]))
    made("no royal", lambda s: [p.update(royal=False) for p in s["pieces"]])
    made("range on a leap", lambda s: s["pieces"][1]["moves"].append(
        {"kind": "leap", "vectors": [[1, 2]], "range": 3}))
    for key, value in (("max_plies", -4), ("no_progress_draw", 0),
                       ("repetition_draw", "often"), ("stalemate", "maybe"), ("bogus", 1)):
        made(f"{key}={value}", lambda s, k=key, v=value: s["rules"].update({k: v}))
    made("wide", lambda s: s["board"].update(width=99))
    made("a twin", lambda s: s["pieces"].append(copy.deepcopy(s["pieces"][0])))
    made("no direction", lambda s: s["pieces"][0].update(moves=[{"kind": "slide"}]))
    made("a warp", lambda s: s["pieces"][0].update(moves=[{"kind": "warp", "vectors": [[0, 1]]}]))
    made("a stranger", lambda s: s["setup"]["placements"].append(
        {"piece": "nope", "squares": ["a1"]}))
    made("a crowd", lambda s: s["setup"]["placements"].append(
        {"piece": s["pieces"][0]["id"], "squares": [s["setup"]["placements"][0]["squares"][0]]}))
    made("a wall off the board", lambda s: s["rules"].update(obstacles={"squares": ["z1"]}))
    made("a walk off the board", lambda s: s["rules"].update(
        npcs=[{"piece": s["pieces"][0]["id"], "route": ["z1"]}]))
    made("a promotion off the board", lambda s: s["rules"].update(
        promotion={"piece": s["pieces"][0]["id"], "rank": 40}))
    # T1 (`docs/archive/plan-pieces-moves.md` §5.3): a path that turns, where it cannot.
    made("a path on a leap", lambda s: s["pieces"][1]["moves"].append(
        {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "path": "circle"}))
    made("a circle with a gap", lambda s: s["pieces"][1]["moves"].append(
        {"kind": "slide", "vectors": [[1, 1]], "symmetry": "lateral", "path": "circle"}))
    made("a zigzag with no turn", lambda s: s["pieces"][1]["moves"].append(
        {"kind": "slide", "vectors": [[0, 1]], "path": "zigzag"}))
    made("a one-step path", lambda s: s["pieces"][1]["moves"].append(
        {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4", "path": "zigzag",
         "range": 1}))
    # A1 (`docs/archive/plan-pieces-moves.md` §6.3): a strike given to some pieces, wrongly.
    def shooting(s: dict[str, Any], by: list[str], **more: Any) -> None:
        royals = [p["id"] for p in s["pieces"] if p.get("royal")]
        s["rules"]["strike"] = {"effect": "capture", "position": "stay", "immune": royals,
                                "by": by, **more}

    made("a strike nobody makes", lambda s: shooting(s, []))
    made("a shooter that never captures", lambda s: (
        s["pieces"][1].update(captures=[]), shooting(s, [s["pieces"][1]["id"]])))
    made("every piece shoots", lambda s: shooting(s, [p["id"] for p in s["pieces"]]))
    made("a shooter beside a side switch", lambda s: shooting(
        s, [s["pieces"][1]["id"]], switches_sides=[s["pieces"][1]["id"]]))
    return out


def _problems() -> list[tuple[str, str, str]]:
    """(where, the validator's sentence, the designer's) for every problem of every draft."""
    found = []
    for name in DRAFTS:
        for broke, draft in _broken(_spec(name)).items():
            # As the designer asks it: settled first, the stamp sentence only when alone.
            draft = designing.settled(draft)
            raw = designing._worth_saying(validate_spec(draft))
            plain = plain_problems(raw, draft)
            assert len(plain) == len(raw), (name, broke)
            found += [(f"{name}: {broke}", one, said) for one, said in zip(raw, plain, strict=True)]
    return found


class TestTheProblemsTheAuditNamed:
    """D48-D50, in the words the audit asked for."""

    def test_a_square_off_the_board_is_named_by_its_row(self) -> None:
        said = plain_problem("setup.placements[0].squares entry 'z9' is not a square on this board")
        assert said == "Where they start, row 1: z9 isn't a square on this board."

    def test_no_royal_piece_and_no_name(self) -> None:
        assert plain_problem("no royal piece appears in setup.placements") == (
            "No royal piece is on the board at the start.")
        assert plain_problem("name must be a non-empty string") == "Give the game a name."
        assert plain_problem("spec is missing required key 'name'") == "Give the game a name."

    def test_how_far_on_a_leap_names_the_piece_and_the_way_of_moving(self) -> None:
        problem = ("pieces[1].moves[0].range means nothing on a leap, which is one jump per "
                   "vector; to reach further, list more vectors, or ride it with a 'slide' "
                   "over the vector or a 'lame' with a range")
        assert plain_problem(problem) == (
            "Piece 2, way of moving 1: 'how far' does nothing on a leap. "
            "To reach further, add more directions, or let it jump again.")
        spec = {"pieces": [{"id": "P", "name": "pawn"}, {"id": "N", "name": "knight"}]}
        assert plain_problem(problem, spec).startswith("Piece 2 (knight), way of moving 1:")

    def test_a_rule_is_named_by_its_cards_title_and_its_clock_in_turns(self) -> None:
        said = plain_problem("rules.max_plies must be between 2 and 10000, got -4")
        assert said == ("The rule 'the longest a game may run' must be between 2 and 10000 "
                        "turns, not -4.")


class TestTheWordsChangedAndTheConditionsDidNot:
    def test_the_validators_own_sentences_are_the_stable_form(self) -> None:
        """The LLM repairs a draft from the path; the designer is shown the translation."""
        draft = _broken(_spec("chess"))["z9"]
        assert "setup.placements[0].squares entry 'z9' is not a square on this board" in (
            validate_spec(draft))
        assert designing.check(draft)["errors"][0] == (
            "Where they start, row 1: z9 isn't a square on this board.")

    def test_every_problem_is_one_sentence_in_the_designers_words(self) -> None:
        found = _problems()
        assert len(found) > 50
        trouble = [f"{where}: {said!r} (from {raw!r})" for where, raw, said in found
                   for phrase in _hits(said) if phrase]
        assert not trouble, "\n".join(trouble)
        for where, _raw, said in found:
            assert said[0].isupper() and said.endswith("."), (where, said)
            assert "_" not in said.replace("'", ""), (where, said)

    def test_a_draft_the_validator_accepts_has_nothing_to_translate(self) -> None:
        for name in DRAFTS:
            assert plain_problems(validate_spec(_spec(name)), _spec(name)) == []

    def test_the_gate_still_refuses_what_it_refused(self) -> None:
        """The thresholds are the gate's, untouched: a game that cannot move is refused,
        and a draft the compiler refuses is refused by name, not raised."""
        spec = _spec("chess")
        stuck = {"name": "stuck", "board": {"width": 3, "height": 3},
                 "pieces": [{"id": "K", "royal": True, "moves": [
                     {"kind": "leap", "vectors": [[0, 1]], "mode": "capture"}]}],
                 "setup": {"mirror": True, "placements": [{"piece": "K", "squares": ["a1"]}]},
                 "rules": {"win": "royal_capture"}}
        assert validate_spec(stuck) == []
        result = gates.smoke_test(stuck, games=10, workers=1)
        assert not result.passed
        assert result.reasons == ["the first player has no legal move from the starting position"]
        broken = copy.deepcopy(spec)
        broken["pieces"][0]["moves"] = [{"kind": "teleport", "vectors": [[1, 1]]}]
        result = gates.smoke_test(broken, games=10, workers=1)
        assert not result.passed and result.reasons[0].startswith(
            "the rules don't fit together: Piece 1 (pawn), way of moving 1:")


class TestTheGatesReasons:
    """D42-D44: every reason the gate can give, read from its source, in a visitor's words."""

    def test_no_reason_holds_a_removed_phrase(self) -> None:
        source = (ROOT / "chess_variant_sandbox" / "loop" / "gates.py").read_text("utf-8")
        reasons = re.findall(r'reasons(?:\.append\(|=\[)\s*((?:f?"[^"]*"\s*)+)', source)
        said = [" ".join(re.findall(r'f?"([^"]*)"', block)) for block in reasons]
        assert len(said) >= 7, said
        trouble = [(text, _hits(text)) for text in said if _hits(text)]
        assert not trouble, trouble

    def test_the_designer_still_finds_the_stall_sentence(self) -> None:
        """`designing` names which counter fired by finding the gate's sentence; the words
        live in one place, so a rewording cannot lose the counter."""
        assert designing._CLOCK_REASON == gates.STALLED
        assert gates.STALLED in (ROOT / "chess_variant_sandbox" / "loop" / "gates.py").read_text(
            "utf-8").split("STALLED =", 1)[1].split("\n", 1)[0]


def _fields(fields: list[dict[str, Any]], path: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for field in fields:
        here = f"{path}.{field['key']}"
        out.append((here, field))
        if field["control"] == "card":
            out += _fields(field["fields"], here)
        elif field["control"] == "list" and field["item"]["control"] == "card":
            out += _fields(field["item"]["fields"], here)
        elif field["control"] == "either":
            for branch in field["branches"]:
                inner = branch["field"]
                if inner["control"] == "card":
                    out += _fields(inner["fields"], here)
                elif inner["control"] == "list" and inner["item"]["control"] == "card":
                    out += _fields(inner["item"]["fields"], here)
    return out


# What a field's sentence may not say: a dotted key, a key's spelling, the machine's yes and
# no, or the text's own vocabulary. A quoted menu value (*'advance'*) is what the menu shows.
_MACHINE = re.compile(
    r"\b[a-z]+_[a-z_]+\b|\b(?:True|False|true|false|ids?|spec|schema|vectors?|validator"
    r"|omit|array|null)\b|\[dx", 0)


class TestTheRuleCardsWords:
    """D40: each field by the designer's name, and a visitor's sentence under it."""

    def _all(self) -> list[tuple[str, dict[str, Any]]]:
        model = cards.card_model()
        return (_fields(model["rules"], "rules") + _fields(model["piece"], "pieces")
                + _fields(model["move"], "pieces.moves")
                + _fields(model["placement"], "setup.placements"))

    def test_every_field_has_the_designers_name(self) -> None:
        unnamed = [path for path, field in self._all()
                   if field["key"] not in ("0", "1") and path not in FIELD_WORDS]
        assert not unnamed, unnamed
        for path, field in self._all():
            assert "_" not in field["label"], (path, field["label"])
            assert not _hits(field["label"]), (path, field["label"])

    def test_no_sentence_under_a_field_speaks_the_texts_language(self) -> None:
        trouble = [(path, field["says"]) for path, field in self._all()
                   if field.get("says") and (_MACHINE.search(field["says"].replace("'", " '"))
                                             or _hits(field["says"]))]
        assert not trouble, "\n".join(f"{p}: {s}" for p, s in trouble)

    def test_a_switch_between_two_spellings_says_what_each_holds(self) -> None:
        labels = {branch["label"] for _, field in self._all() if field["control"] == "either"
                  for branch in field["branches"]}
        assert not labels & {"integer", "string", "boolean", "number"}, labels
        assert cards._branch_label({"type": "integer"}) == KIND_WORDS["integer"] == "a number"
        assert cards._branch_label({"type": "boolean"}) == "yes or no"


class TestTheSmallChangesNameTheRule:
    """M7: the map's *small changes to the rules* read the rule, never its key."""

    @pytest.mark.parametrize("name", DRAFTS)
    def test_no_step_sentence_holds_a_key(self, name: str) -> None:
        # One case a draft (Q997b: all three in one test cost 4-5 s); the claim holds for
        # each draft's sentences alone, and each draft must itself reach the turn-limit rule.
        said: set[str] = set()
        spec = _spec(name)
        for operator in OPERATORS:
            for seed in range(12):
                child = copy.deepcopy(spec)
                line = operator.apply(child, random.Random(seed))
                if line:
                    said.add(line)
        assert any("the turn limit" in line or "the no-progress rule" in line for line in said)
        trouble = [(line, _hits(line)) for line in said
                   if _hits(line) or re.search(r"\b[a-z]+_[a-z_]+\b", line)]
        assert not trouble, trouble


class TestTheDesignersOwnSentencesAboutRulesAndEndings:
    """D45-D46: a random game's ending in turns and in words, a rule by its card's title,
    and the game in between refused in the rules' terms."""

    def test_an_ending_is_said_in_words_and_never_by_its_key(self) -> None:
        class Ended:
            def __init__(self, reason: str) -> None:
                self.reason = reason

        assert designing._reason(Ended("ply_limit")) == "by the turn limit"
        assert designing._reason(Ended("no_progress")) == "by the no-progress rule"
        assert designing._reason(Ended("a_reason_from_later")) == "by the rules"
        assert designing._rule_title("must_capture") == "a capture must be taken"

    def test_a_random_game_that_did_not_finish_is_counted_in_turns(self) -> None:
        said = designing.unfinished(_spec("chess"))["says"]
        assert " turns" in said, said
        assert not _hits(said) and "_" not in said, said

    def test_a_rule_the_draft_does_not_have_is_named_by_its_title(self) -> None:
        with pytest.raises(designing.DesignError, match="no 'walls on the board' rule"):
            designing.demonstrate(_spec("chess"), "obstacles")


class TestTheScreensReasonsAreAVisitors:
    """Q681: the middle rung's two refusals reach the designer on both servers
    (`site/screening.py`, `web/candidacy.screen_draft`), so they speak of a *computer
    player* and of *turns*, never a *searching player*, *plies* or a budget."""

    def test_both_refusals_hold_no_phrase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib
        from types import SimpleNamespace

        # The package exports the function under the module's name.
        screening = importlib.import_module("chess_variant_sandbox.loop.screen")

        def played(winner: Any, plies: int) -> Any:
            records = [SimpleNamespace(winner=winner, plies=plies) for _ in range(8)]
            monkeypatch.setattr(screening, "run_tournament",
                                lambda *_, **__: SimpleNamespace(records=records))
            return screening.screen({})

        drawn, trapped = played(None, 200), played(0, 4)
        assert not drawn.passed and not trapped.passed
        said = [*drawn.reasons, *trapped.reasons]
        assert len(said) == 2, said
        assert "no computer player won any of 8 games" in said[0]
        assert "within 12 turns" in said[1]
        for reason in said:
            assert not _hits(reason), (reason, _hits(reason))
            assert "budget" not in reason, reason


# ------------------------------------------------ Q922: the survivors of the mutation report

_TWO_PIECES: dict[str, Any] = {
    "pieces": [{"id": "A", "name": "ant"}, {"id": "B"}],
    "setup": {"placements": []},
    "rules": {},
}

# A problem the validator can say about a draft, and the designer's sentence for it. The
# paths reach one step past the end of what the draft holds, or stop one step short of a
# further index, because those are the edges a path reader has to survive.
PLAIN_SENTENCES = {
    "a piece one past the end of the list": (
        "pieces[2] must be an object", "Piece 3 is written wrongly."),
    "a piece far past the end, with a field": (
        "pieces[5].moves must be a list", "Piece 6: 'ways of moving' must be a list."),
    "a named piece's field": (
        "pieces[0].moves must be a list", "Piece 1 (ant): 'ways of moving' must be a list."),
    "a field with no index on its way of moving": (
        "pieces[1].moves must be a list", "Piece 2: 'ways of moving' must be a list."),
    "a way of moving, by its number": (
        "pieces[1].moves[2].kind must be one of 'leap', 'slide'",
        "Piece 2, way of moving 3: 'how it travels' must be one of 'leap', 'slide'."),
    "the placements list itself": (
        "setup.placements must be a list", "Where they start must be a list."),
    "one placement, whole": (
        "setup.placements[0] must be an object", "Where they start, row 1 is written wrongly."),
    "one placement's field, by its row": (
        "setup.placements[3].squares must be a list",
        "Where they start, row 4: 'squares' must be a list."),
    "the setup, whole": ("setup must be an object", "Where they start is written wrongly."),
    "a setup field": (
        "setup.mirror must be a boolean",
        "Where they start: 'mirror for the other player' must be yes or no."),
    "the second of several promotion rules": (
        "rules.promotion[1].to must be a list",
        "The rule 'promotion', number 2: 'it may become' must be a list."),
    "an index with no number": (
        "rules.promotion[].to must be a list",
        "The rule 'promotion': 'it may become' must be a list."),
    "the first of several promotion rules, whole": (
        "rules.promotion[0] must be an object",
        "The rule 'promotion', number 1 is written wrongly."),
    "an index inside a field already named": (
        "rules.turn.ramp[2] must be a positive integer",
        "The rule 'more than one move a turn': 'the opening turns' "
        "must be a whole number, 1 or more."),
    "a rule that opens a colon": (
        "rules.max_plies: oops", "The rule 'the longest a game may run': oops."),
    "a rule's value in quotes, no field": (
        "rules.win 'foo' is not allowed",
        "The rule 'how the game is won' 'foo' is not allowed."),
    "a menu's values as the menu spells them": (
        "rules.win must be one of 'checkmate', 'royal_capture'",
        "The rule 'how the game is won' must be one of 'checkmate', 'royal capture'."),
    "a bare word that is also a path root": (
        "rules.max_plies must include the name of rules",
        "The rule 'the longest a game may run' must include the name of rules."),
    "no path at all": ("something odd happened", "Something odd happened."),
    "a path run into the next character": (
        "pieces[0]x must be fine", "Piece 1 (ant)x must be fine."),
    "a path followed by a comma": (
        "board.width, must be", "The board, 'board width', must be."),
    "a missing key the words know": (
        "pieces[0] is missing required key 'moves'", "Piece 1 (ant): 'ways of moving' is missing."),
    "a missing key the words do not know": (
        "pieces[0] is missing required key 'foo_bar'", "Piece 1 (ant): 'foo bar' is missing."),
}


@pytest.mark.parametrize("label", sorted(PLAIN_SENTENCES))
def test_a_problem_is_said_in_the_designers_words(label: str) -> None:
    problem, said = PLAIN_SENTENCES[label]
    assert plain_problem(problem, _TWO_PIECES) == said


class TestAPieceFieldNamesItsPiecesForEveryReader:
    """Q995e. A piece field the validator names with no index (`pieces[].speed`) was named
    where it is only in the designer (Q995b). `spec.plain` does it now, so the command line
    and the proposing model's repairs read the same sentence."""

    @staticmethod
    def _speeding() -> dict[str, Any]:
        spec = _spec("chess")
        spec["pieces"][1]["speed"] = 1
        return spec

    def test_where_reads_a_field_of_the_army(self) -> None:
        from chess_variant_sandbox.spec.plain import _where

        spec = self._speeding()
        assert _where("pieces[].speed", spec) == ("Piece 2 (knight)", "speed")
        assert _where("pieces.speed", spec) == ("Piece 2 (knight)", "speed")
        spec["pieces"][2]["speed"] = 1
        assert _where("pieces[].speed", spec)[0] == "Pieces 2 (knight) and 3 (bishop)"
        assert _where("pieces[].spy", spec)[0] == "Every piece"

    def test_the_sentence_names_the_piece_and_says_where_rounds_are_chosen(self) -> None:
        spec = self._speeding()
        line = next(p for p in validate_spec(spec) if p.startswith("pieces[].speed"))
        said = plain_problem(line, spec)
        assert said.startswith("Piece 2 (knight): a speed only matters"), said
        assert "how the players take turns" in said

    def test_the_designer_says_what_spec_plain_says(self) -> None:
        spec = self._speeding()
        for line in validate_spec(spec):
            assert designing.plain_problem(line, spec) == plain_problem(line, spec), line

    def test_the_command_line_refusal_names_the_piece(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        from chess_variant_sandbox import cli

        path = tmp_path / "speedy.json"
        path.write_text(json.dumps(self._speeding()), encoding="utf-8")
        assert cli.main(["smoke", str(path), "--games", "10", "--workers", "1"]) == 1
        out = capsys.readouterr().out
        assert "Piece 2 (knight): a speed only matters in a game played in rounds" in out, out
        assert "Where the game comes from" not in out


# ------------------------------------------------ Q1013: a command's words, one source of them


class TestACommandsWords:
    """Q1013 (`docs/archive/plan-schema-31.md` §3): `spec.plain` words the fourth sight effect and its
    `captures_own` as it words freeze, disarm and shield, and the designer's sight rows read
    those words and no stand-in of their own (Q1007d's, gone). Pinned on a game whose knights
    command (K1, no turncoat capture) and one whose king commands with `captures_own` (K2)."""

    def test_the_effect_and_its_box_are_worded_beside_the_other_three(self) -> None:
        assert FIELD_WORDS["pieces.sight.command"] == ("commands what it sees", None)
        assert FIELD_WORDS["pieces.sight.captures_own"] == (
            "may make them take their own side",
            "Only beside a command. Yes: the other player may also make a commanded piece take "
            "a piece of its own side that it may capture, never a royal or a shielded one, and "
            "not while it is disarmed. Left empty, a commanded piece moves only onto empty "
            "squares.")
        said = FIELD_WORDS["pieces.sight.effect"][1]
        assert said is not None
        assert said.endswith(
            "Shield: nothing may capture it, and a shielded royal is never in check. Command: "
            "the other player may also move it, as its owner would, onto an empty square; a "
            "royal is never commanded, and a frozen piece is moved by nobody.")
        for words in (FIELD_WORDS["pieces.sight.command"], FIELD_WORDS["pieces.sight.captures_own"],
                      FIELD_WORDS["pieces.sight.effect"]):
            for text in words:
                assert text is None or not (_MACHINE.search(text.replace("'", " '"))
                                            or _hits(text)), text

    def test_the_sight_rows_read_spec_plain_alone(self) -> None:
        assert not hasattr(cards, "SIGHT_WORDS_PENDING") and not hasattr(cards, "COMMAND_SENTENCE")
        words = cards.card_model()["sight"]["words"]
        for key in cards.SIGHT_FIELDS:
            label, says = FIELD_WORDS[f"pieces.sight.{key}"]
            assert words[key] == [label, says or ""], key

    def test_a_command_game_and_a_turncoat_one_are_accepted(self) -> None:
        from test_command import k1, k2

        for spec in (k1(), k2()):
            assert plain_problems(validate_spec(spec), spec) == []

    def test_each_refusal_of_a_command_is_said_in_the_designers_words(self) -> None:
        from test_command import KING_COMMAND, KNIGHT_COMMAND, chess, k1, k2, with_sight

        reach = k1()
        reach["rules"]["reach"] = {"squares": ["e4"]}
        hand = k2()
        hand["rules"]["drops"] = {}
        no_turncoat = with_sight(chess(), "N", {**KNIGHT_COMMAND, "affects": ["B"],
                                                "captures_own": True})
        next(p for p in no_turncoat["pieces"] if p["id"] == "B")["captures"] = ["K"]
        cases = {
            "beside a reach": (reach, (
                "A command ('what it sees' effect 'command') beside the rule 'winning by "
                "arriving': whether a commanded arrival wins for the mover or the owner would be "
                "a second rule; leave one out.")),
            "a turncoat beside a capture hand": (hand, (
                "A command's 'may make them take their own side' beside the rule 'pieces played "
                "back onto the board', 'captured pieces': whose hand a turncoat capture fills "
                "would be a second rule; leave one out.")),
            "a turncoat beside a freeze": (
                with_sight(chess(), "N", {**KNIGHT_COMMAND, "effect": "freeze",
                                          "captures_own": True}),
                "Piece 2 (knight), line of sight 1: 'may make them take their own side' means "
                "something only beside effect 'command' (a commanded piece made to take its own "
                "side); leave it out."),
            "a turncoat written wrongly": (
                with_sight(chess(), "K", {**KING_COMMAND, "captures_own": "yes"}),
                "Piece 6 (king), line of sight 1: 'may make them take their own side' must be "
                "true or false."),
            "a turncoat nothing could make": (no_turncoat, (
                "Piece 2 (knight), line of sight 1: affects nothing that could be there: a "
                "command's 'may make them take their own side' where no type it commands may "
                "take a piece of its own side.")),
        }
        for name, (spec, sentence) in cases.items():
            said = plain_problems(validate_spec(spec), spec)
            assert sentence in said, (name, said)
            for line in said:
                assert line[0].isupper() and line.endswith("."), (name, line)
                assert "_" not in line.replace("'", ""), (name, line)
                assert not _hits(line), (name, line)

    def test_no_step_sentence_on_a_command_game_holds_a_key(self) -> None:
        from test_command import k1, k2

        said: set[str] = set()
        for spec in (k1(), k2()):
            for operator in OPERATORS:
                for seed in range(6):
                    child = copy.deepcopy(spec)
                    line = operator.apply(child, random.Random(seed))
                    if line:
                        said.add(line)
        assert any("command" in line for line in said), said
        trouble = [(line, _hits(line)) for line in said
                   if _hits(line) or re.search(r"\b[a-z]+_[a-z_]+\b", line)]
        assert not trouble, trouble


class TestTheRulesArePlural:
    """LP0 (`docs/archive/plan-landing.md` §5). The validator's *the spec* is *the rules* to a
    visitor, and the verb after it stayed singular: *the rules states no ending of its
    own*. The fix is in the substitution (`plain.the_rules`), so every sentence built the
    same way agrees, and the refusal itself is unchanged: same problem, one for one."""

    @pytest.mark.parametrize(("spec_words", "rules_words"), [
        ("the spec states no ending of its own", "the rules state no ending of their own"),
        ("the spec is", "the rules are"),
        ("the spec was", "the rules were"),
        ("the spec has", "the rules have"),
        ("the spec does", "the rules do"),
        ("the spec says", "the rules say"),
        ("the spec carries", "the rules carry"),
        ("the spec passes", "the rules pass"),
        ("the spec already states", "the rules already state"),
        ("the spec alone decides", "the rules alone decide"),
        ("the spec itself says", "the rules themselves say"),
        ("the spec's own copy", "the rules' own copy"),
        ("the spec as written", "the rules as written"),
        ("the spec stated", "the rules stated"),
        ("the spec, then", "the rules, then"),
    ])
    def test_what_agrees_with_the_rules_turns_plural(
            self, spec_words: str, rules_words: str) -> None:
        from chess_variant_sandbox.spec.plain import the_rules

        assert the_rules(spec_words) == rules_words

    def test_the_unwinnable_refusal_reads_right_in_the_designer(self) -> None:
        spec = _spec("chess")
        spec["rules"]["win"] = "none"
        problems = validate_spec(spec)
        line = next(p for p in problems if p.startswith("rules.win is 'none'"))
        assert "the spec states" in line  # the validator's own sentence is unchanged
        expected = ("The rule 'how the game is won' is 'none' but the rules state no ending "
                    "of their own; the game could never be won.")
        assert plain_problem(line, spec) == expected
        said = designing.check(spec)["errors"]
        assert expected in said and len(said) == len(problems)

    def test_no_validator_sentence_says_the_rules_with_a_singular_verb(self) -> None:
        """Every literal sentence in the validator's sources that says *the spec*, through
        the designer's words: none leaves *the rules* with a singular verb."""
        singular = re.compile(
            r"\bthe rules (?:already |alone |also )?(?:is|was|has|does|[a-z]+[^s']s)\b")
        # A sentence's own line: a string argument, never a docstring.
        sentence = re.compile(r'^\s*f?"(?!"")([^"\n]*\bthe spec\b[^"\n]*)"', re.M)
        found = []
        for path in sorted((ROOT / "chess_variant_sandbox" / "spec").glob("*.py")):
            for literal in sentence.findall(path.read_text("utf-8")):
                said = plain_problem(f"rules.win {literal}")
                found.append(said)
                assert not singular.search(said), (path.name, said)
        assert found
