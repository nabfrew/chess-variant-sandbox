"""Pins `measurement.py`'s source-fingerprint readers and helpers to exact outputs.

The Rust token reading (`_ident_byte`, `_rust_token`, `_test_module_end`,
`_rust_token_reading`, `_rust_reading`) is pinned token by token on small inline source
texts that hold every token kind and the edges of the input, and the small helpers beside
it (`fingerprint_workers`) on each side of their bounds. Expected values are literals,
checked by hand against what Rust means by each text.
"""

from __future__ import annotations

import json

import pytest

from chess_variant_sandbox import measurement

# Every token kind: line, nested block and doc-free comments; plain, escaped, raw (with and
# without hashes), byte-raw and C-raw strings; char literals (escaped, two-byte UTF-8, NUL);
# lifetimes and a label; `r` after an identifier byte (`x`, `A`, `_`), which is not raw;
# CR, LF, tab, vertical tab and form feed.
_RICH = (
    b"fn f<'a>(x: &'a [u8]) -> u8 { // tail\n"
    b"\t/* one /* two */ three */ let s = \"q\\\"//\";\r\n"
    b"  let r = r#\"a\"b\"#; let b = br\"k\"; let c = cr##\"j\"#\"##;\x0b\x0c"
    b"let e = '\\''; let u = '\xc3\xa9'; let z = '\x00'; xr\"m\"; Ar\"n\"; _r\"o\"; 'lbl: loop {} 'x'"
)


def _tokens(src: bytes) -> list[tuple[str, bytes]]:
    out = []
    index = 0
    while index < len(src):
        kind, end = measurement._rust_token(src, index)
        assert end > index
        out.append((kind, src[index:end]))
        index = end
    return out


def test_ident_byte_is_exactly_the_rust_identifier_bytes():
    ident = {byte for byte in range(256) if measurement._ident_byte(byte)}
    expected = (
        set(range(0x30, 0x3A)) | set(range(0x41, 0x5B)) | {0x5F}
        | set(range(0x61, 0x7B)) | set(range(0x80, 0x100))
    )
    assert ident == expected
    # The edges on each side, spelled out.
    assert [measurement._ident_byte(b) for b in (0x2F, 0x30, 0x39, 0x3A, 0x40, 0x41, 0x5A, 0x5B,
                                                0x5E, 0x5F, 0x60, 0x61, 0x7A, 0x7B, 0x7F, 0x80)] == [
        False, True, True, False, False, True, True, False,
        False, True, False, True, True, False, False, True,
    ]


def test_rust_token_reads_a_rich_source_token_by_token():
    tokens = _tokens(_RICH)
    kinds = "".join(kind[0] for kind, _ in tokens)
    assert kinds == (
        "bbsbbbbbbbbsbbbsbbbbbsbbsbbsbsc"
        "sscsbbbsbsbslbssssbbbsbsbslbsbbbsbsbslbsbbbsbsbslbss"
        "bbbsbsbslbsbbbsbsbslbsbbbsbsbslbsbblbsbblbsbblbsbbbbbsbbbbsbbsl"
    )
    assert [(kind, text) for kind, text in tokens if kind in ("comment", "literal")] == [
        ("comment", b"// tail"),
        ("comment", b"/* one /* two */ three */"),
        ("literal", b'"q\\"//"'),
        ("literal", b'r#"a"b"#'),
        ("literal", b'br"k"'),
        ("literal", b'cr##"j"#"##'),
        ("literal", b"'\\''"),
        ("literal", b"'\xc3\xa9'"),
        ("literal", b"'\x00'"),
        ("literal", b'"m"'),
        ("literal", b'"n"'),
        ("literal", b'"o"'),
        ("literal", b"'x'"),
    ]
    assert b"".join(text for _, text in tokens) == _RICH


@pytest.mark.parametrize(
    ("src", "index", "expected"),
    [
        (b"x", 0, ("byte", 1)),  # the last byte has no following byte
        (b"'", 0, ("byte", 1)),  # a lone quote at the end is not a char literal
        (b"r", 0, ("byte", 1)),
        (b"a'", 1, ("byte", 2)),
        (b"// x", 0, ("comment", 4)),  # a line comment that runs to the end of the file
        (b"// x\ny", 0, ("comment", 4)),
        (b'xr"a"', 1, ("byte", 2)),  # `r` after an identifier byte is not a raw string
        (b'(r"a")', 1, ("literal", 5)),
        (b"'\x00'", 0, ("literal", 3)),  # a NUL char literal
        (b"'\\n'", 0, ("literal", 4)),
        (b" ", 0, ("space", 1)),
    ],
)
def test_rust_token_at_the_edges(src, index, expected):
    assert measurement._rust_token(src, index) == expected


@pytest.mark.parametrize("src", [b"/* open", b"/* a /* b */", b'"open', b'r#"open"', b"'\\nope"])
def test_rust_token_refuses_an_unterminated_literal_or_comment(src):
    with pytest.raises(measurement._Unterminated):
        measurement._rust_token(src, 0)


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (b" mod tests {\n fn a() { '}' }\n}\nfn b() {}", 30),
        (b"  mod   _t9{ } x", 14),
        (b" mod {x}", None),  # no name: a brace where the name should be
        (b" mod 9a {}", None),  # a digit first
        (b" mod t;", None),
        (b" mod \xc3\xa9 {}", None),  # a name past ASCII
        (b"mod{}", None),  # no space before the name
        (b" fn t {}", None),
    ],
)
def test_test_module_end(src, expected):
    assert measurement._test_module_end(src, 0) == expected


def test_test_module_end_refuses_an_unclosed_module():
    with pytest.raises(measurement._Unterminated):
        measurement._test_module_end(b" mod t {", 0)


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            _RICH,
            b"fn f<'a>(x: &'a [u8]) -> u8 { let s = \"q\\\"//\"; let r = r#\"a\"b\"#; "
            b"let b = br\"k\"; let c = cr##\"j\"#\"##; let e = '\\''; let u = '\xc3\xa9'; "
            b"let z = '\x00'; xr\"m\"; Ar\"n\"; _r\"o\"; 'lbl: loop {} 'x'",
        ),
        (
            b"fn a() {}\n#[cfg(test)]\nmod t { #[test] fn x() { let c = '}'; } }\nfn b() {}",
            b"fn a() {} fn b() {}",
        ),
        (b"#[cfg(test)] mod t;\n fn c() {}", b"#[cfg(test)] mod t; fn c() {}"),
        (b"x // end", b"x"),
        (b"  \n\t", b""),
    ],
)
def test_rust_token_reading(src, expected):
    assert measurement._rust_token_reading(src) == expected


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (b"a\r\nb", b"a b"),
        (b"  a /* open", b"  a /* open"),  # in doubt: the LF bytes
        (b'a "open', b'a "open'),
    ],
)
def test_rust_reading(src, expected):
    assert measurement._rust_reading(src) == expected


@pytest.mark.parametrize(("cores", "expected"), [(None, 1), (1, 1), (2, 1), (3, 2), (5, 4), (64, 4)])
def test_fingerprint_workers_default_is_the_cores_less_one_at_most_four(monkeypatch, cores, expected):
    import os

    monkeypatch.delenv(measurement.FINGERPRINT_WORKERS_ENV, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    assert measurement.fingerprint_workers() == expected


@pytest.mark.parametrize(("asked", "expected"), [("x", 2), ("0", 2), ("-3", 2), ("1", 1), ("9", 9)])
def test_fingerprint_workers_reads_the_variable(monkeypatch, asked, expected):
    import os

    monkeypatch.setenv(measurement.FINGERPRINT_WORKERS_ENV, asked)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert measurement.fingerprint_workers() == expected


def test_utf8_width_reads_every_lead_byte():
    widths = [measurement._utf8_width(lead) for lead in range(256)]
    assert widths == [1] * 0xC0 + [2] * 0x20 + [3] * 0x10 + [4] * 0x10


@pytest.mark.parametrize(
    ("src", "index", "expected"),
    [
        # A nested opener whose `*` starts a closer: `/*/` opens, it does not close.
        (b"/* /*/ */ */", 0, ("comment", 12)),
        (b"'\xe2\x82\xac'", 0, ("literal", 5)),  # a three-byte UTF-8 char literal
        (b"'\xf0\x9f\x98\x80'", 0, ("literal", 6)),  # a four-byte one
    ],
)
def test_rust_token_nested_comments_and_wide_chars(src, index, expected):
    assert measurement._rust_token(src, index) == expected


def test_test_module_end_stops_a_name_at_the_first_byte_past_ascii():
    assert measurement._test_module_end(b" mod t\x80 {}", 0) is None
    assert measurement._test_module_end(b" mod t\x7f {}", 0) is None


@pytest.mark.parametrize(
    ("body", "expected"),
    [(b"", 0), (b"ab", 2), (b"ab\ncd", 5), (b"ab\ncd\nef", 5), (b"\nab\ncd", 3),
     (b"\n", 1), (b"\n\n", 1), (b"\n\nx", 1)],
)
def test_second_line_end(body, expected):
    assert measurement._second_line_end(body) == expected


@pytest.mark.parametrize(
    ("agents", "shown", "expected"),
    [
        ([], 2, "nothing"),
        (["a"], 2, "a"),
        (["a", "b"], 2, "a, b"),
        (["a", "b", "c"], 2, "a, b and 1 more"),
        (["a", "b", "c", "d", "e"], 3, "a, b, c and 2 more"),
    ],
)
def test_first_names_two_and_counts_the_rest(agents, shown, expected):
    assert measurement._first(agents, shown) == expected


def test_package_imports_names_every_module_and_its_parents():
    body = (
        b"import os, chess_variant_sandbox.metrics.index as ix\n"
        b"from chess_variant_sandbox.spec import canonical, schema  # note\n"
        b"from .seeds import (combine,\n    verdict)\n"
        b"from .. import trust\n"
        b"import json; from ...outside import z\n"
    )
    assert measurement.package_imports(body, ("chess_variant_sandbox", "metrics")) == {
        "chess_variant_sandbox",
        "chess_variant_sandbox.metrics",
        "chess_variant_sandbox.metrics.index",
        "chess_variant_sandbox.spec",
        "chess_variant_sandbox.spec.canonical",
        "chess_variant_sandbox.spec.schema",
        "chess_variant_sandbox.metrics.seeds",
        "chess_variant_sandbox.metrics.seeds.combine",
        "chess_variant_sandbox.metrics.seeds.verdict",
        "chess_variant_sandbox.trust",
    }


def test_closure_follows_imports_and_the_entries_own_packages(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    files = {
        "__init__.py": "",  # above the package: no entry's import reaches it
        "chess_variant_sandbox/__init__.py": "",
        "chess_variant_sandbox/top/__init__.py": "",
        "chess_variant_sandbox/top/a.py": "from chess_variant_sandbox.low import b\n",
        "chess_variant_sandbox/low/__init__.py": "",
        "chess_variant_sandbox/low/b.py": "from .c import f\n",
        "chess_variant_sandbox/low/c.py": "def f():\n    pass\n",
        "chess_variant_sandbox/low/d.py": "",  # imported by nothing
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(measurement, "ROOT", root)
    found = measurement._closure(("chess_variant_sandbox/top/a.py",))
    assert {path.relative_to(root).as_posix() for path in found} == {
        "chess_variant_sandbox/__init__.py",
        "chess_variant_sandbox/top/__init__.py",
        "chess_variant_sandbox/top/a.py",
        "chess_variant_sandbox/low/__init__.py",
        "chess_variant_sandbox/low/b.py",
        "chess_variant_sandbox/low/c.py",
    }


def test_values_source_key_is_cached_for_the_process(tmp_path, monkeypatch):
    import hashlib

    other = tmp_path / "x.py"
    other.write_bytes(b"x = 1\n")
    measurement.values_source_key.cache_clear()
    try:
        monkeypatch.setattr(measurement, "values_sources", lambda: [])
        first = measurement.values_source_key()
        monkeypatch.setattr(measurement, "values_sources", lambda: [other])
        assert measurement.values_source_key() == first
        assert first == hashlib.sha256(b"values-cache 1\0").hexdigest()
    finally:
        measurement.values_source_key.cache_clear()


def test_read_disk_reuses_what_it_read_only_for_the_same_path_and_key(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv(measurement.TABLES_CACHE_ENV, str(tmp_path / "elsewhere.json"))
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    for path in (one, two):
        path.write_text(json.dumps({"sources": "k", "tables": {"on": "disk"}}), encoding="utf-8")
    state = {"path": one, "key": "k", "tables": {"in": "memory"}}
    # The same path and key: what was read, and the file is not read again.
    assert measurement._read_disk(state, one, "k") == {"in": "memory"}
    # Another path under the same key: that file's answers, and the state follows it.
    assert measurement._read_disk(state, two, "k") == {"on": "disk"}
    assert state == {"path": two, "key": "k", "tables": {"on": "disk"}}


@pytest.mark.parametrize(
    ("stored", "now", "reason"),
    [
        ("a", "b", "its nets were a, and b play now"),
        ("", "b", "its nets were none, and b play now"),
        ("a", "", "its nets were a, and none play now"),
    ],
)
def test_stale_against_names_the_nets(stored, now, reason):
    base = {
        "ruleset_hash": "abc123", "weights_version": "v8", "agents": ("random", "greedy"),
        "games_per_pairing": 4, "seeds": (0,), "engine": "e", "metrics": "m",
        "tables": "t", "analysis": "a",
    }
    staleness = measurement.Basis(**base, net=stored).stale_against(
        measurement.Basis(**base, net=now)
    )
    assert staleness.reasons == (reason,)
    assert staleness.needs == "replay"


def test_staleness_is_frozen_and_hashable():
    import dataclasses

    staleness = measurement.Staleness(reasons=("r",), needs="rescore")
    assert hash(staleness) == hash(measurement.Staleness(reasons=("r",), needs="rescore"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        staleness.needs = ""  # type: ignore[misc]


@pytest.mark.parametrize(("src", "end"), [(b'r""', 3), (b'r#""#', 5), (b'br"" x', 4)])
def test_rust_token_reads_an_empty_raw_string(src, end):
    # The closing quote may be the byte right after the opening one.
    assert measurement._rust_token(src, 0) == ("literal", end)


@pytest.mark.parametrize("src", [b" mod t", b" mod t  ", b"\nmod abc\t"])
def test_test_module_end_answers_none_when_the_source_ends_after_the_name(src):
    assert measurement._test_module_end(src, 0) is None


@pytest.mark.parametrize("src", [b'"a"', b'let s = "a"', b"x b'q' r\"z\""])
def test_rust_token_reading_keeps_a_literal_that_ends_the_source(src):
    assert measurement._rust_token_reading(src) == src


def test_source_key_reads_an_unreadable_file_as_a_marker(tmp_path):
    import hashlib

    missing = tmp_path / "gone.rs"
    expected = hashlib.sha256(
        b"lbl\0" + str(missing).encode("utf-8") + b"\0<unreadable>\0"
    ).hexdigest()
    assert measurement._source_key("lbl", [missing]) == expected


def test_stale_build_speaks_only_when_the_sources_are_here_and_differ(monkeypatch):
    monkeypatch.setattr(measurement, "built_fingerprint", lambda: "built")
    monkeypatch.setattr(measurement, "rust_source_fingerprint", lambda: None)
    assert measurement.stale_build() is None
    monkeypatch.setattr(measurement, "rust_source_fingerprint", lambda: "built")
    assert measurement.stale_build() is None
    monkeypatch.setattr(measurement, "rust_source_fingerprint", lambda: "other")
    assert measurement.stale_build() == (
        "the loaded core was built from different sources than rust/src/ "
        "-- rebuild with `uv sync --extra dev --reinstall-package chess-variant-sandbox`"
    )


def test_basis_repr_leaves_out_the_bytes_and_seed_sets():
    basis = measurement.Basis(
        ruleset_hash="abc123", weights_version="v8", agents=("random",),
        games_per_pairing=4, seeds=(0,), engine="e", metrics="m",
        engine_bytes="EB1", metrics_bytes="MB2", analysis_bytes="AB3",
        known_seed_sets=((7, 8),),
    )
    shown = repr(basis)
    assert "ruleset_hash='abc123'" in shown
    for hidden in ("engine_bytes", "EB1", "metrics_bytes", "MB2", "analysis_bytes", "AB3", "known_seed_sets"):
        assert hidden not in shown


def test_hash_files_skips_a_file_it_cannot_read(tmp_path):
    present = tmp_path / "a.py"
    present.write_bytes(b"x = 1\n")
    missing = tmp_path / "b.py"
    # FNV-1a over nothing is the offset basis.
    assert measurement._hash_files([missing]) == "cbf29ce484222325"
    assert measurement._hash_files([present, missing]) == measurement._hash_files([present])


def test_rust_token_reads_a_nested_comment_opened_right_after_the_outer_one():
    # `/*/` opens one comment and then a second at the `/` it shares with the first.
    assert measurement._rust_token(b"/*/**/*/x", 0) == ("comment", 8)


@pytest.mark.parametrize("src", [b" mod ", b"\tmod\n", b" mod  \t"])
def test_test_module_end_answers_none_when_the_source_ends_after_mod(src):
    assert measurement._test_module_end(src, 0) is None


def test_rust_token_reading_spaces_a_test_module_between_two_bytes():
    assert measurement._rust_token_reading(b"a#[cfg(test)] mod t {}b") == b"a b"
    assert measurement._rust_token_reading(b"#[cfg(test)] mod t {}b") == b"b"


def test_code_reading_cuts_a_docstring_and_a_comment_on_one_line():
    body = b'def f():\n    """doc"""  # c\n    return 1\n'
    assert measurement._code_reading(body) == b"def f():\n    return 1"


def test_bytes_fingerprint_is_cached_for_the_process(tmp_path, monkeypatch):
    other = tmp_path / "x.py"
    other.write_bytes(b"x = 1\n")
    measurement.bytes_fingerprint.cache_clear()
    try:
        monkeypatch.setattr(measurement, "_sources", lambda *patterns: [])
        first = measurement.bytes_fingerprint("metrics")
        monkeypatch.setattr(measurement, "_sources", lambda *patterns: [other])
        assert measurement.bytes_fingerprint("metrics") == first == "cbf29ce484222325"
    finally:
        measurement.bytes_fingerprint.cache_clear()


def test_warm_fingerprints_compiles_a_spec_whose_values_are_cold(tmp_path, monkeypatch):
    import concurrent.futures

    from chess_variant_sandbox.spec import canonical

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    monkeypatch.setattr(canonical, "spec_hash", lambda spec: spec["id"])
    monkeypatch.setattr(measurement, "tables_cache_path", lambda: tmp_path / "t.json")
    monkeypatch.setattr(measurement, "values_cache_path", lambda: tmp_path / "v.json")
    monkeypatch.setattr(measurement, "compile_source_key", lambda: "kt")
    monkeypatch.setattr(measurement, "values_source_key", lambda: "kv")
    monkeypatch.setattr(measurement, "_read_disk", lambda state, path, key: {})
    written = []
    monkeypatch.setattr(
        measurement, "_write_disk_many",
        lambda state, path, key, answers: written.append((path.name, key, dict(answers))),
    )
    monkeypatch.setattr(measurement, "_cold_pair", lambda job: (job[1], "T" + job[1], "V" + job[1]))
    tables = {"A": "t", "B": "t", "C": "t", "D": "t"}
    values = {"D": "v"}
    monkeypatch.setattr(measurement, "_TABLES", tables)
    monkeypatch.setattr(measurement, "_VALUES", values)
    # A, B and C have tables but no values, so they are cold; D has both and is not.
    specs = [{"id": name} for name in "ABCD"]
    assert measurement.warm_fingerprints(specs, workers=2) == 3
    assert values == {"A": "VA", "B": "VB", "C": "VC", "D": "v"}
    assert written == [
        ("t.json", "kt", {"A": "TA", "B": "TB", "C": "TC"}),
        ("v.json", "kv", {"A": "VA", "B": "VB", "C": "VC"}),
    ]


def test_net_players_marks_a_net_it_cannot_read(monkeypatch):
    from chess_variant_sandbox.agents import nets

    monkeypatch.setattr(measurement, "_net_procedures", lambda agents: ["p"])

    def refuse(options, ruleset_hash):
        raise OSError("gone")

    monkeypatch.setattr(nets, "net_bytes", refuse)
    assert measurement.net_players(("minimax:net=p",), "abc") == "p=unreadable"


def test_staleness_counts_no_stored_runs_unless_told():
    staleness = measurement.Staleness(
        needs="replay", missing_seeds=(1000,), stored_needs="rescore", expected_runs=4
    )
    assert staleness.stored_runs == 0
    assert staleness.cost() == "replay 1 of 4 runs"
    assert measurement.Staleness(
        needs="replay", missing_seeds=(1000,), stored_needs="rescore", expected_runs=4,
        stored_runs=3,
    ).cost() == "replay 1 of 4 runs, re-score 3"


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (b"//", ("comment", 2)),  # the second byte is the last: still a comment
        (b"x//", ("byte", 1)),
        (b"'a", ("byte", 1)),  # a `'` whose closing quote would fall past the end
        (b"'a'", ("literal", 3)),
    ],
)
def test_rust_token_reads_a_construct_that_ends_the_file(src, expected):
    assert measurement._rust_token(src, 0) == expected


def test_first_shows_two_names_by_default():
    assert measurement._first(["a", "b", "c"]) == "a, b and 1 more"
    assert measurement._first(["a", "b"]) == "a, b"


def test_package_imports_reads_on_past_a_relative_import_above_the_package():
    body = b"from ....far import x\nfrom chess_variant_sandbox.spec import schema\n"
    assert measurement.package_imports(body, ("chess_variant_sandbox", "metrics")) == {
        "chess_variant_sandbox",
        "chess_variant_sandbox.spec",
        "chess_variant_sandbox.spec.schema",
    }


@pytest.mark.parametrize("value", [True, 2.0, "3"])
def test_certificate_refuses_a_count_that_is_not_a_plain_int(value):
    with pytest.raises(ValueError, match="a certificate's sample is a positive count"):
        measurement.Certificate(sample=value, games=4, engine="e", tables="t")


def test_compute_values_is_sixteen_hex_of_the_values_json(monkeypatch):
    from chess_variant_sandbox.tournament import play

    monkeypatch.setattr(measurement, "_compiled", lambda spec, digest: "compiled")
    monkeypatch.setattr(play, "eval_values", lambda compiled: [1.0, 2.5])
    assert measurement._compute_values({}, "d") == "a16498ae00524512"


def test_players_fingerprint_is_sixteen_hex_of_the_seats_and_values(monkeypatch):
    import chess_variant_sandbox.agents as agents

    spec = json.loads((measurement.ROOT / "rulesets" / "dull.json").read_text())
    monkeypatch.setattr(agents, "native_spec", lambda agent, digest="": ("rnd", (1, 2), b"net"))
    monkeypatch.setattr(measurement, "values_fingerprint", lambda spec: "v16")
    assert measurement.players_fingerprint(spec, ("random",)) == "a46dbe8bdc16104a"


def test_built_fingerprint_is_read_once_for_the_process(monkeypatch):
    import chess_variant_sandbox._core as core

    measurement.built_fingerprint.cache_clear()
    try:
        first = measurement.built_fingerprint()
        assert first == str(getattr(core, "BUILD_FINGERPRINT", "unknown"))
        # The loaded extension cannot change under a running process: the first read holds.
        monkeypatch.setattr(core, "BUILD_FINGERPRINT", "moved", raising=False)
        assert measurement.built_fingerprint() == first
    finally:
        measurement.built_fingerprint.cache_clear()


def test_compile_source_key_is_cached_for_the_process(tmp_path, monkeypatch):
    import hashlib

    other = tmp_path / "x.py"
    other.write_bytes(b"x = 1\n")
    measurement.compile_source_key.cache_clear()
    try:
        monkeypatch.setattr(measurement, "compile_sources", lambda: [])
        first = measurement.compile_source_key()
        monkeypatch.setattr(measurement, "compile_sources", lambda: [other])
        assert measurement.compile_source_key() == first
        assert first == hashlib.sha256(b"tables-cache 1\0").hexdigest()
    finally:
        measurement.compile_source_key.cache_clear()


def test_module_file_without_listings_finds_a_module_and_not_an_attribute():
    package = measurement.ROOT / "chess_variant_sandbox"
    assert measurement._module_file("chess_variant_sandbox.measurement") == package / "measurement.py"
    assert measurement._module_file("chess_variant_sandbox") == package / "__init__.py"
    assert measurement._module_file("chess_variant_sandbox.measurement.Basis") is None


def test_package_of_a_file_outside_the_root_is_empty(tmp_path):
    assert measurement._package_of(tmp_path / "x.py") == ()
    inside = (measurement.ROOT / "chess_variant_sandbox" / "spec" / "x.py").resolve()
    assert measurement._package_of(inside) == ("chess_variant_sandbox", "spec")


def test_certificate_and_basis_are_frozen_and_hashable():
    import dataclasses

    cert = measurement.Certificate(sample=3, games=4, engine="e", tables="t")
    assert hash(cert) == hash(measurement.Certificate(sample=3, games=4, engine="e", tables="t"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cert.sample = 5  # type: ignore[misc]
    fields = {
        "ruleset_hash": "abc123", "weights_version": "v8", "agents": ("random",),
        "games_per_pairing": 4, "seeds": (0,), "engine": "e", "metrics": "m",
    }
    basis = measurement.Basis(**fields)
    assert hash(basis) == hash(measurement.Basis(**fields))
    with pytest.raises(dataclasses.FrozenInstanceError):
        basis.engine = "other"  # type: ignore[misc]


def test_stale_against_compares_the_verdict_rule_from_two_seeds():
    base = {
        "ruleset_hash": "abc123", "weights_version": "v8", "agents": ("random", "greedy"),
        "games_per_pairing": 4, "seeds": (0, 1), "engine": "e", "metrics": "m",
        "tables": "t", "analysis": "a",
    }
    staleness = measurement.Basis(**base, verdict_rule="rule-a").stale_against(
        measurement.Basis(**base, verdict_rule="rule-b")
    )
    assert staleness.reasons == ("its verdict was taken under rule-a, not rule-b",)
    assert staleness.needs == "rescore"


def test_measurement_save_writes_two_space_json(tmp_path):
    stored = measurement.Measurement(
        hash="abc123", directory=tmp_path, stored={"score": 1.5},
        basis=None, staleness=measurement.UNKNOWN,
    )
    path = tmp_path / "m.json"
    stored.save(path)
    assert path.read_text(encoding="utf-8") == '{\n  "score": 1.5,\n  "basis": null\n}'


def test_test_module_end_refuses_a_name_that_starts_with_any_digit():
    # `0` is the smallest digit: a module named `0t` is not a Rust identifier.
    assert measurement._test_module_end(b" mod 0t {}", 0) is None
    assert measurement._test_module_end(b" mod 9t {}", 0) is None
    assert measurement._test_module_end(b" mod t0 {}", 0) == 10


def test_built_fingerprint_reads_unbuilt_when_the_core_will_not_import(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "chess_variant_sandbox._core", None)
    assert measurement.built_fingerprint.__wrapped__() == "unbuilt"


def test_closure_keeps_an_entry_it_cannot_read(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "chess_variant_sandbox").mkdir()
    (root / "chess_variant_sandbox" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(measurement, "ROOT", root)
    found = measurement._closure(("chess_variant_sandbox/gone.py",))
    assert {path.relative_to(root).as_posix() for path in found} == {
        "chess_variant_sandbox/__init__.py",
        "chess_variant_sandbox/gone.py",
    }


def test_main_runs_dir_follows_a_relative_gitdir_pointer(tmp_path, monkeypatch):
    main = tmp_path / "main"
    admin = main / ".git" / "worktrees" / "w"
    admin.mkdir(parents=True)
    (admin / "commondir").write_text("../..\n", encoding="utf-8")
    root = tmp_path / "w"
    root.mkdir()
    (root / ".git").write_text("gitdir: ../main/.git/worktrees/w\n", encoding="utf-8")
    monkeypatch.setattr(measurement, "ROOT", root)
    assert measurement.main_runs_dir() == main.resolve() / "runs"


def test_warm_fingerprints_skips_a_repeated_spec_and_keeps_reading(tmp_path, monkeypatch):
    import concurrent.futures
    from pathlib import Path

    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.spec.canonical import spec_hash

    rulesets = Path(__file__).resolve().parents[1] / "rulesets"
    first, second, third = (
        load_spec(rulesets / f"{name}.json") for name in ("dull", "losalamos", "gardner")
    )
    monkeypatch.setenv(measurement.TABLES_CACHE_ENV, str(tmp_path / "cache" / "t.json"))
    monkeypatch.setattr(measurement, "_TABLES", {})
    monkeypatch.setattr(measurement, "_VALUES", {})
    monkeypatch.setattr(measurement, "_DISK", {"path": None, "key": None, "tables": {}})
    monkeypatch.setattr(measurement, "_VALUES_DISK", {"path": None, "key": None, "tables": {}})
    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(
        measurement, "_cold_pair", lambda job: (job[1], "t-" + job[1], "v-" + job[1])
    )
    # The repeat comes first: skipping it must not end the reading of the rest.
    assert measurement.warm_fingerprints([first, first, second, third], workers=2) == 3
    digests = [spec_hash(spec) for spec in (first, second, third)]
    assert {d: "t-" + d for d in digests} == measurement._TABLES
    assert {d: "v-" + d for d in digests} == measurement._VALUES


def test_hash_files_reads_on_past_an_unreadable_first_file(tmp_path):
    missing = tmp_path / "a.py"  # sorts first, so the skip must not end the reading
    present = tmp_path / "b.py"
    present.write_bytes(b"x = 1\n")
    assert measurement._hash_files([missing, present]) == measurement._hash_files([present])
    assert measurement._hash_files([missing, present]) != "cbf29ce484222325"


def test_store_swallows_a_directory_it_cannot_make(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    measurement._store(blocker / "cache.json", "k", {"a": "b"})  # an OSError, swallowed
    assert blocker.read_text(encoding="utf-8") == "a file, not a directory"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["blocker"]


def _stub_warm(monkeypatch, tmp_path, tables, values, pool=None):
    """`warm_fingerprints` with every cache and the compiler stubbed; returns the compile calls."""
    import concurrent.futures

    from chess_variant_sandbox.spec import canonical

    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", pool or concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(canonical, "spec_hash", lambda spec: spec["id"])
    monkeypatch.setattr(measurement, "tables_cache_path", lambda: tmp_path / "t.json")
    monkeypatch.setattr(measurement, "values_cache_path", lambda: tmp_path / "v.json")
    monkeypatch.setattr(measurement, "compile_source_key", lambda: "kt")
    monkeypatch.setattr(measurement, "values_source_key", lambda: "kv")
    monkeypatch.setattr(measurement, "_read_disk", lambda state, path, key: {})
    monkeypatch.setattr(measurement, "_write_disk_many", lambda state, path, key, answers: None)
    compiled: list[str] = []

    def cold_pair(job):
        compiled.append(job[1])
        return job[1], "T" + job[1], "V" + job[1]

    monkeypatch.setattr(measurement, "_cold_pair", cold_pair)
    monkeypatch.setattr(measurement, "_TABLES", tables)
    monkeypatch.setattr(measurement, "_VALUES", values)
    return compiled


def test_warm_fingerprints_reads_on_past_a_warm_spec(tmp_path, monkeypatch):
    compiled = _stub_warm(monkeypatch, tmp_path, {"W": "t"}, {"W": "v"})
    # The warm spec comes first: skipping it must not end the reading of the rest.
    specs = [{"id": name} for name in "WABC"]
    assert measurement.warm_fingerprints(specs, workers=2) == 3
    assert sorted(compiled) == ["A", "B", "C"]


def test_warm_fingerprints_leaves_two_cold_specs_to_the_lazy_path(tmp_path, monkeypatch):
    compiled = _stub_warm(monkeypatch, tmp_path, {}, {})
    assert measurement.warm_fingerprints([{"id": "A"}, {"id": "B"}], workers=2) == 0
    assert compiled == []
    assert measurement.warm_fingerprints([{"id": n} for n in "ABC"], workers=2) == 3


def test_warm_fingerprints_survives_a_pool_that_will_not_start(tmp_path, monkeypatch):
    def no_pool(*args, **kwargs):
        raise RuntimeError("daemonic processes are not allowed to have children")

    tables: dict[str, str] = {}
    compiled = _stub_warm(monkeypatch, tmp_path, tables, {}, pool=no_pool)
    assert measurement.warm_fingerprints([{"id": n} for n in "ABC"], workers=2) == 0
    assert compiled == [] and tables == {}


def test_warm_fingerprints_reads_on_past_a_spec_that_will_not_compile(tmp_path, monkeypatch):
    tables: dict[str, str] = {}
    _stub_warm(monkeypatch, tmp_path, tables, {})

    def cold_pair(job):
        if job[1] == "A":
            raise ValueError("a spec the compiler rejects")
        return job[1], "T" + job[1], "V" + job[1]

    monkeypatch.setattr(measurement, "_cold_pair", cold_pair)
    # The failing spec comes first: its failure must not end the reading of the rest.
    assert measurement.warm_fingerprints([{"id": n} for n in "ABC"], workers=2) == 2
    assert tables == {"B": "TB", "C": "TC"}


def test_package_imports_reads_on_past_a_from_import_with_no_module():
    body = b"from  import x\nfrom chess_variant_sandbox.spec import schema\n"
    assert measurement.package_imports(body, ("chess_variant_sandbox", "metrics")) == {
        "chess_variant_sandbox",
        "chess_variant_sandbox.spec",
        "chess_variant_sandbox.spec.schema",
    }


def test_closure_reads_on_past_an_entry_it_cannot_read(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    package = root / "chess_variant_sandbox"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(measurement, "ROOT", root)
    # The missing entry is read before `a.py`: its failure must not end the walk.
    found = measurement._closure(("chess_variant_sandbox/a.py", "chess_variant_sandbox/gone.py"))
    assert {path.relative_to(root).as_posix() for path in found} == {
        "chess_variant_sandbox/__init__.py",
        "chess_variant_sandbox/a.py",
        "chess_variant_sandbox/gone.py",
    }


def test_code_reading_keeps_whitespace_only_on_the_lines_a_string_spans():
    body = b'x = """a  \n   \nb"""  \n\ny = 1  # c\n'
    assert measurement._code_reading(body) == b'x = """a  \n   \nb"""\ny = 1'


def test_code_fingerprint_is_cached_for_the_process(tmp_path, monkeypatch):
    path = tmp_path / "x.py"
    path.write_bytes(b"x = 1\n")
    monkeypatch.setattr(measurement, "_sources", lambda *patterns: [path])
    measurement.code_fingerprint.cache_clear()
    try:
        first = measurement.code_fingerprint("pattern")
        path.write_bytes(b"x = 2\n")  # a code change, which the reading would see
        assert measurement.code_fingerprint("pattern") == first
        measurement.code_fingerprint.cache_clear()
        assert measurement.code_fingerprint("pattern") != first
    finally:
        measurement.code_fingerprint.cache_clear()


def test_rust_source_fingerprint_is_cached_for_the_process(tmp_path, monkeypatch):
    monkeypatch.setattr(measurement, "ROOT", tmp_path)
    measurement.rust_source_fingerprint.cache_clear()
    try:
        assert measurement.rust_source_fingerprint() is None
        (tmp_path / "rust" / "src").mkdir(parents=True)
        (tmp_path / "rust" / "src" / "a.rs").write_bytes(b"fn a() {}\n")
        assert measurement.rust_source_fingerprint() is None  # the first answer holds
        measurement.rust_source_fingerprint.cache_clear()
        assert measurement.rust_source_fingerprint() is not None
    finally:
        measurement.rust_source_fingerprint.cache_clear()


def test_seeded_gives_nothing_without_a_main_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv(measurement.TABLES_CACHE_ENV, raising=False)
    monkeypatch.setattr(measurement, "main_runs_dir", lambda: None)
    assert measurement._seeded(tmp_path / "t.json", "k") == {}


@pytest.mark.parametrize("checked", ["sample:", "sample:x", "sample:-2", "sample: 3"])
def test_read_check_refuses_a_sample_without_a_plain_count(checked):
    assert measurement.read_check(checked) is None


def test_measurement_is_frozen(tmp_path):
    import dataclasses

    stored = measurement.Measurement(
        hash="abc123", directory=tmp_path, stored={}, basis=None, staleness=measurement.UNKNOWN,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        stored.hash = "other"  # type: ignore[misc]
