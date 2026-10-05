//! Bake a fingerprint of the core's sources into the core itself.
//!
//! A measurement is a function of the code that took it, and the one thing the harness
//! could never see was which code that was. A score taken before stage 17C and one
//! taken after are different quantities -- 58 of 58 corpus entries played differently
//! across that change -- and nothing on disk recorded the difference, so a stale number
//! and a current one looked identical.
//!
//! Hashing `rust/src/` from Python would answer "what does the source say now". It
//! would not answer "what is the extension that is actually loaded", which is the
//! question behind the footgun `CLAUDE.md` sets in bold: a plain `uv sync` does not
//! notice that `rust/` changed, so you can edit the core, run the tests, and be testing
//! the previous build. Comparing the fingerprint compiled in here against the same hash
//! taken from the source on disk is what turns that into an error message.
//!
//! FNV-1a rather than a real digest so this needs no build-dependency, and because
//! `chess_variant_sandbox/measurement.py` has to reproduce it byte for byte -- a short, fully
//! specified hash is easier to keep honest across two languages than a vendored SHA.
//! Collision resistance buys nothing here; the question is only ever "same or not".
//!
//! What is hashed is the *token reading* of each file's LF form, never the bytes on disk:
//! comments, whitespace runs and `#[cfg(test)]` modules out, literals kept byte for byte
//! (Q630, `token_reading.rs`). See `fold_crlf` for the LF half, and `.gitattributes` for
//! the other half of that repair.

#[path = "token_reading.rs"]
mod token_reading;

use std::fs;
use std::path::PathBuf;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn absorb(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// Fold CRLF to LF, so a checkout's line endings cannot move the fingerprint.
///
/// `core.autocrlf=true` on this machine and the committed blobs are LF, so a fresh
/// `.claude/worktrees/` checkout holds CRLF where the main checkout holds LF for the same
/// commit. Both fingerprints hashed raw bytes, so identical code fingerprinted differently
/// in the two checkouts and a campaign run in a worktree could never be delivered as
/// current -- `docs/archive/bugs-resolved.md` has the 60-spec measurement. Hashing the LF reading makes the
/// number blind to the difference; `.gitattributes` stops the drift arising at all.
///
/// A lone `\r` is left alone. Git's conversion only ever adds or removes the *pair*, so
/// folding a bare CR would buy nothing and would collapse two files that genuinely differ
/// -- a fingerprint that fails to notice a change, which is the one direction that is
/// never safe. LF-only input is untouched by construction, so no LF fingerprint moves.
///
/// `_hash_files` in `chess_variant_sandbox/measurement.py` folds the same way on the same bytes, and
/// `tests/test_measurement.py::TestTheBuildFingerprint` holds the two readings together.
fn fold_crlf(bytes: &[u8]) -> Vec<u8> {
    let mut folded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        index += 1;
        if byte == b'\r' && bytes.get(index) == Some(&b'\n') {
            continue;
        }
        folded.push(byte);
    }
    folded
}

fn main() {
    let mut sources: Vec<PathBuf> = fs::read_dir("src")
        .expect("rust/src must exist to build the core")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .collect();
    // Sorted by file name, so the fingerprint is a property of the sources rather than
    // of the order the filesystem happened to hand them over in.
    sources.sort();

    let mut hash = FNV_OFFSET;
    for path in &sources {
        let name = path
            .file_name()
            .expect("a file yielded by read_dir has a name")
            .to_string_lossy()
            .into_owned();
        let body = fs::read(path).expect("a file yielded by read_dir can be read");
        absorb(&mut hash, name.as_bytes());
        absorb(&mut hash, b"\0");
        let folded = fold_crlf(&body);
        // Anything in doubt -- a literal or comment that runs off the end -- is read as
        // its LF bytes, the more sensitive reading, as `measurement._rust_reading` does.
        let reading = token_reading::token_reading(&folded).unwrap_or(folded);
        absorb(&mut hash, &reading);
        absorb(&mut hash, b"\0");
        println!("cargo:rerun-if-changed={}", path.display());
    }

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=token_reading.rs");
    println!("cargo:rustc-env=CHESS_VARIANT_SANDBOX_BUILD_FINGERPRINT={hash:016x}");
}
