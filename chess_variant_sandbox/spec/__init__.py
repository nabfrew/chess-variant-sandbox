"""The rule DSL: schema, validation, canonical serialisation, content hashing.

Rule specs are plain JSON data. Nothing in this package executes spec content; the
engine compiles a validated spec into lookup tables. See CLAUDE.md.
"""

from chess_variant_sandbox.spec.canonical import canonical_json, load_spec, spec_hash
from chess_variant_sandbox.spec.complexity import SpecComplexity, measure_complexity
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, SpecError, validate_spec

__all__ = [
    "SCHEMA_VERSION",
    "SpecComplexity",
    "SpecError",
    "canonical_json",
    "load_spec",
    "measure_complexity",
    "spec_hash",
    "validate_spec",
]
