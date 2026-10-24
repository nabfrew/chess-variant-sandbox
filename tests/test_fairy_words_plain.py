"""`fairy/words.py` says "the spec" as "the rules" through `spec.plain`, with no table of its own."""

from __future__ import annotations

import inspect

from chess_variant_sandbox.fairy import words
from chess_variant_sandbox.spec.plain import the_rules


def test_the_spec_substitution_is_plains() -> None:
    assert words.the_rules is the_rules
    assert words.plain_reason("the spec does not say") == "the rules do not say"


def test_words_holds_no_spec_substitution_of_its_own() -> None:
    assert "the spec" not in inspect.getsource(words.plain_reason)
