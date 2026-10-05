//! `build.rs`'s token reading against `reading-corpus/`, the expected readings that
//! `tests/test_measurement.py` holds `measurement._rust_reading` to as well. Both sides
//! agreeing with one hand-written list is what keeps the build fingerprint the same
//! number in two languages.

#[path = "../token_reading.rs"]
mod token_reading;

use std::fs;
use std::path::Path;

#[test]
fn every_corpus_case_reads_as_written() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("reading-corpus");
    let mut cases = 0;
    for entry in fs::read_dir(&corpus).expect("rust/reading-corpus exists") {
        let path = entry.expect("a corpus entry").path();
        if path.extension().is_none_or(|ext| ext != "in") {
            continue;
        }
        let source = fs::read(&path).expect("a corpus input");
        let expected = fs::read(path.with_extension("out")).expect("each input has an .out");
        // As `build.rs` hashes it: the reading, or the bytes when one is in doubt.
        let reading = token_reading::token_reading(&source).unwrap_or(source);
        assert_eq!(
            String::from_utf8_lossy(&reading),
            String::from_utf8_lossy(&expected),
            "{}",
            path.display()
        );
        cases += 1;
    }
    assert!(cases >= 15, "only {cases} corpus cases found");
}

#[test]
fn every_core_source_reads_as_code() {
    // No real source falls back to its bytes, or a comment there would still move the
    // fingerprint; and every one loses something, or the reading is not being applied.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for entry in fs::read_dir(&src).expect("rust/src exists") {
        let path = entry.expect("a source entry").path();
        if path.extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        let body = fs::read(&path).expect("a source");
        let reading = token_reading::token_reading(&body)
            .unwrap_or_else(|| panic!("{} reads as bytes", path.display()));
        assert!(reading.len() < body.len(), "{}", path.display());
    }
}
