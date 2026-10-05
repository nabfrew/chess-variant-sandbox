//! The token reading of a Rust source file: what the build fingerprint hashes (Q630).
//!
//! Shared by `build.rs`, which bakes the fingerprint into the core, and by
//! `tests/token_reading.rs`, which pins it against `reading-corpus/`.
//! `chess_variant_sandbox/measurement.py::_rust_token_reading` is the same function in
//! Python, and `tests/test_measurement.py` pins it against the same corpus, so the two
//! sides agree byte for byte or a test fails.
//!
//! Exactly: every comment (line, doc and nested block) and every `#[cfg(test)] mod name
//! { ... }` goes; each run of whitespace, comments and test modules outside a literal
//! becomes one space, none at either end; every other byte, and every literal (string,
//! byte string, C string, raw string, char) byte for byte, stays. A literal or block
//! comment that runs off the end answers `None`, and the caller hashes the LF bytes.
//!
//! Rust is whitespace-insensitive outside literals, doc comments become `#[doc]`
//! attributes nothing reads, and `rust/src/` uses none of `line!`, `file!`, `column!`,
//! `#[track_caller]` or `include_str!`. So nothing this drops can change what the release
//! build does, and a comment, a reflow or a test-only edit leaves the fingerprint alone.

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Space,
    Comment,
    Literal,
    Byte,
}

const TEST_MODULE: &[u8] = b"#[cfg(test)]";

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn ident_byte(byte: u8) -> bool {
    byte >= 0x80 || byte == b'_' || byte.is_ascii_alphanumeric()
}

fn utf8_width(lead: u8) -> usize {
    if lead >= 0xF0 {
        4
    } else if lead >= 0xE0 {
        3
    } else if lead >= 0xC0 {
        2
    } else {
        1
    }
}

fn find(src: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > src.len() {
        return None;
    }
    src[from..]
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|offset| from + offset)
}

fn string_end(src: &[u8], mut index: usize) -> Option<usize> {
    while index < src.len() {
        match src[index] {
            b'\\' => index += 2,
            b'"' => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}

/// The token at `index`: its kind and end, or `None` when it runs off the end.
fn token(src: &[u8], index: usize) -> Option<(Kind, usize)> {
    let n = src.len();
    let byte = src[index];
    if is_space(byte) {
        return Some((Kind::Space, index + 1));
    }
    let following = src.get(index + 1).copied();
    if byte == b'/' && following == Some(b'/') {
        return Some((Kind::Comment, find(src, b"\n", index).unwrap_or(n)));
    }
    if byte == b'/' && following == Some(b'*') {
        let (mut depth, mut at) = (1usize, index + 2);
        while depth > 0 {
            if at >= n {
                return None;
            }
            if src[at..].starts_with(b"/*") {
                depth += 1;
                at += 2;
            } else if src[at..].starts_with(b"*/") {
                depth -= 1;
                at += 2;
            } else {
                at += 1;
            }
        }
        return Some((Kind::Comment, at));
    }
    if byte == b'"' {
        return string_end(src, index + 1).map(|end| (Kind::Literal, end));
    }
    if index == 0 || !ident_byte(src[index - 1]) {
        // A raw string: `r`, `br` or `cr`, any number of `#`, then `"`.
        let at = if (byte == b'b' || byte == b'c') && following == Some(b'r') {
            index + 1
        } else {
            index
        };
        if src[at] == b'r' {
            let mut quote = at + 1;
            while quote < n && src[quote] == b'#' {
                quote += 1;
            }
            if quote < n && src[quote] == b'"' {
                let mut close = vec![b'"'];
                close.extend(std::iter::repeat_n(b'#', quote - at - 1));
                let end = find(src, &close, quote + 1)?;
                return Some((Kind::Literal, end + close.len()));
            }
        }
    }
    if byte == b'\'' {
        if let Some(next) = following {
            if next == b'\\' {
                let end = find(src, b"'", index + 3)?;
                return Some((Kind::Literal, end + 1));
            }
            let close = index + 1 + utf8_width(next);
            if close < n && src[close] == b'\'' {
                return Some((Kind::Literal, close + 1));
            }
        }
    }
    Some((Kind::Byte, index + 1))
}

/// Where `#[cfg(test)] mod name { ... }` ends, if that is what starts at `index` (just
/// past the attribute). Anything else answers `Some(None)` and is read as code.
fn test_module_end(src: &[u8], index: usize) -> Option<Option<usize>> {
    let n = src.len();
    let spaces = |mut at: usize| {
        while at < n && is_space(src[at]) {
            at += 1;
        }
        at
    };
    let mut at = spaces(index);
    if !src[at..].starts_with(b"mod") {
        return Some(None);
    }
    at += 3;
    let name = spaces(at);
    if name == at || name >= n || src[name] >= 0x80 || !ident_byte(src[name])
        || src[name].is_ascii_digit()
    {
        return Some(None);
    }
    at = name;
    while at < n && src[at] < 0x80 && ident_byte(src[at]) {
        at += 1;
    }
    at = spaces(at);
    if at >= n || src[at] != b'{' {
        return Some(None);
    }
    let mut depth = 0usize;
    while at < n {
        let (kind, end) = token(src, at)?;
        if kind == Kind::Byte && src[at] == b'{' {
            depth += 1;
        } else if kind == Kind::Byte && src[at] == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(Some(end));
            }
        }
        at = end;
    }
    None
}

/// The token reading of an LF source, or `None` when a literal or comment is unterminated.
pub fn token_reading(src: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(src.len());
    let mut pending = false;
    let mut index = 0;
    while index < src.len() {
        if src[index..].starts_with(TEST_MODULE) {
            if let Some(end) = test_module_end(src, index + TEST_MODULE.len())? {
                index = end;
                pending = true;
                continue;
            }
        }
        let (kind, end) = token(src, index)?;
        if kind == Kind::Space || kind == Kind::Comment {
            pending = true;
        } else {
            if pending && !out.is_empty() {
                out.push(b' ');
            }
            pending = false;
            out.extend_from_slice(&src[index..end]);
        }
        index = end;
    }
    Some(out)
}
