"""Schema 9, the Python half: a piece that neither player owns.

`rules.npcs` is specified in `docs/archive/plan-schema-6-9.md` stage C. What is asserted
here is the half that exists before the core can play it: the validator refuses what it
says it refuses,
the compiler builds the tables the core will read, `bridge.rust_supports` holds the
intermediate state shut, every surface learned the rule -- and, first, **the bump moved
nothing**. The fingerprints below were computed on the commit before this one and match
the bases stored under the main checkout's `runs/`, 51 of 51 (the 52nd, `hierarchy-extended`,
landed beside the bump and is frozen from the schema-9 compiler; its row says why that is
the same number); a moved fingerprint here is a ruleset whose compiled tables changed, and
no ruleset's should have.

One kind of row is no longer of that kind, and it says so in place. A ruleset whose *spec* is
edited moves both of its numbers on purpose, and then the stored basis is the one that is
behind rather than the table -- `intransitive` is that row, from the user's `c947850`.

Schema 9's own showcase had four rows here and no longer does: Q67 removed `archive/boar` and
its three arms on 2026-09-07, so **no ruleset in the tree uses `rules.npcs`**. The frozen
pairs are kept in a comment beside where they stood, for a rebuilt showcase to be compared
against.

The core half is `tests/test_schema_nine_core.py`'s: the phase, the undo log, check from a
wild piece, the tally, and what `perft` counts. It landed in Q9b, and the guard section
below is what is left of the refusal window here.
"""

from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox import fuzz
from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.agents.features import (
    VOCABULARY,
    FeatureEvaluator,
    _score_gap,
    feature_context,
)
from chess_variant_sandbox.engine import Engine, bridge
from chess_variant_sandbox.engine.bridge import canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.measurement import tables_fingerprint
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec

ROOT = Path(__file__).resolve().parents[1]

# `(mechanics hash, tables fingerprint)` for every committed and archived ruleset, taken
# on the commit before schema 9's compiler change. Both numbers are pinned: the hash says
# the file is the one that was measured, the fingerprint says the compiler still produces
# what it produced then. A new ruleset gets a row here when it is added -- the coverage
# test below says so by name -- and a fingerprint that moves is a regression until a bump
# argues otherwise, in which case the argument is written down and the row re-frozen.
#
# **47 of the 60 fingerprints were re-frozen on 2026-09-07 by Q113, and the argument is
# `docs/reference/derived-irreversibility.md`.** The boundary gained one key,
# `one_way_types`: which pieces can never return to a square they left, derived from the
# movement tables, replacing an irreversible-move test that read `piece_type == ep_type`
# and so made *a pawn move is progress* depend on whether the game happened to declare en
# passant. No mechanics hash moved -- no spec file was edited -- and the 13 rows that did
# not move are exactly the rulesets with no one-way piece at all, where the new key sits
# at its inert default and is dropped. **Fifteen of the 47 also play differently**; the
# other 32 gained a fingerprint for a field that says something true about them and
# changes no game, which is the price of the field existing and is named in the costing.
FROZEN: dict[str, tuple[str, str]] = {
    "rulesets/almost.json": ("9c26a04f1ca5", "b5693955ab65e13a"),
    "rulesets/antichess.json": ("3ec1560124c3", "fe591ce72b1ac298"),
    "rulesets/atomic.json": ("6bd4d9c5769e", "4c0a2bc790ce84a3"),
    "rulesets/bastion.json": ("9574b0bf9095", "c7b2c37ce93eaaa3"),
    "rulesets/berolina.json": ("14a126b38a1a", "f2b419020c29d309"),
    "rulesets/capablanca.json": ("2eefcd835b3b", "304b04afb9d2d47b"),
    # Re-frozen 2026-09-07 by Q98, on the same footing as `intransitive` below: the
    # phantom `"en_passant": true` came off the spec when `_check_inert_rules` learned
    # to refuse an en passant with no double step to answer (the user's decision,
    # `docs/archive/decisions.md`, 2026-09-07). Not a compiler regression -- a different
    # game, so both numbers moved together (`86efa5819135`/`e8793d50c5ebf20b` before).
    # The tables move because `en_passant` is a compiled field, even though `ep_passes`
    # was empty both sides of the edit, which is the whole reason the rule was inert.
    # `runs/86efa5819135-0` is a measurement of the spec as it was and is deliberately
    # kept; measuring the spec as it is now is Q103.
    # Q869, 2026-10-13: the attack sources keep only the frames a type can stand in, so a
    # piece that faces no way loses its sources for the facings it never holds; the same
    # games move for move (played and perft-compared both ways). was 711c14261a57f2dc
    "rulesets/chess-oriented-pawns.json": ("57c73975c905", "d3841caf15b539ee"),
    "rulesets/chess.json": ("4c1922ffae26", "94a4a49ab238a4e2"),
    "rulesets/courier.json": ("0b96463ffba7", "43c2fe4fb1ab1401"),
    # Q869, 2026-10-13: the rear pawn `V` respelled as a pawn placed facing backwards with a
    # facing no move turns (schema 25), so the hand holds one kind of pawn and returns it
    # facing forward: a new game, `runs/ab865f3f954a-*` kept. was ab865f3f954a / db15f09d67ddd2d1
    "rulesets/crazyhouse-torus.json": ("9598e603241d", "5842b620af27402b"),
    "rulesets/crazyhouse.json": ("bb5d97520084", "97ab14ffca01037a"),
    "rulesets/cylinder.json": ("a9306688fe57", "6a56433c2492eaf3"),
    "rulesets/dobutsu.json": ("a2f0c65ceedf", "d158e949aa940a34"),
    "rulesets/dull.json": ("fe7c42772c71", "0909393f145e303c"),
    "rulesets/gardner.json": ("975dde173f01", "c70b9b68c5e7108a"),
    "rulesets/horde.json": ("15593fc47f6d", "abcf4aed8c6ec0b9"),
    # Re-frozen twice, and both times because the spec was wrong about its own game
    # rather than because the compiler moved. 2026-09-06, the user's own `c947850`
    # ("The territory rule in intransitive as an error.") deleted `rules.territory`:
    # `15e3eb35bfbc`/`9d7931bf5dfb17fd` -> `09861015bb3a`/`8c802f78403e2edc`. Then
    # 2026-09-07, Q120 added `rules.repetition_draw: "never"` under a `schema_version`
    # of 10, because the key was absent and therefore read as the default 3 where the
    # designer states there is no draw by repetition. Both numbers move together each
    # time: a different game, not a regression. The compiled tables move only in
    # `repetition_draw` (3 -> `u32::MAX`) and in the five Zobrist draws that any hash
    # change re-seeds. `runs/15e3eb35bfbc-0` and `runs/09861015bb3a-0` are kept under the
    # main checkout as measurements of the two specs that no longer exist.
    # Re-frozen a third time 2026-10-04 by Q696, on purpose: the user's decision of
    # 2026-10-03 moved `rules.no_progress_draw` from 100 to 200 plies, the reading of
    # the program the designer's official site plays. Was `d93e28d09a33` /
    # `d6fd4ae715ff1843`; its runs (`runs/d93e28d09a33` and `-0`, `-1000`, `-2000`,
    # `-3000`) are kept as the measurement of a spec that no longer exists, and
    # `runs/29d5213b02b2` is the current one.
    "rulesets/intransitive.json": ("29d5213b02b2", "44343f5d0820d0f8"),
    "rulesets/kingofthehill.json": ("71912ea8ba7f", "58ed70e1e5ca6fd5"),
    "rulesets/knightmate.json": ("50a3da8f75e7", "09bf8280bead8d65"),
    "rulesets/losalamos.json": ("fcbb53d53ce4", "48e1a3d58f9c6bd5"),
    "rulesets/makruk.json": ("eaa48ed29ad7", "fc1b479eb1a32a55"),
    "rulesets/onewaycylinder.json": ("b02bbe8da72c", "0449ce95d1a9ac76"),
    "rulesets/onewaytorus.json": ("c6ff883e5919", "05787a033547e413"),
    # Q868, 2026-10-12: `rulesets/onewaytunnel.json` (`da7647bf02e1`/`fac342b6c6256f3c`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    "rulesets/racingkings.json": ("66b001f1b790", "48c162212370a86a"),
    "rulesets/rps-hierarchy-chess.json": ("c0e0c35f78b1", "cee254479fe50d0f"),
    "rulesets/rps-v1.json": ("2261f8203948", "b079ab1b18c454a6"),
    "rulesets/rps-v2.json": ("82a4788c02f0", "096cd759f1426837"),
    # Re-frozen 2026-10-04 by Q708 (F1): the game is unchanged (same hash); its tables gain
    # `goal_distance_each`, the per-goal race table the seal reads, sent only where a priced
    # race has a side with several goal squares. It was "2ba31ac7f701533d".
    "rulesets/rps-v3.json": ("bdf2801ade24", "05becb3282c4dcbe"),
    "rulesets/rps-v4.json": ("add543cc28c3", "5e6ef0f6db086829"),
    "rulesets/rps-v5.json": ("a7b6f22e6b73", "32c2dec25152fa8b"),
    # The sardines family's wrap cell, Q180, 2026-09-11. It replaces
    # `sardines-wraparound.json` (`f76f73bd5d6c`/`551032910084b2d6`), whose rank seam was
    # one-way per side; this one is the full torus, so the rename carries a new game and
    # not only a new name. `runs/f76f73bd5d6c-0` stays as the old game's measurement.
    # Q868, 2026-10-12: `rulesets/sardines-torus.json` (`f5585062433b`/`6f3802cc1cdb323a`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Re-frozen 2026-09-11 by Q180, on the `stones` / `chess-oriented-pawns` footing: a
    # spec replaced on purpose, so both numbers move together and it is a different game
    # rather than a compiler regression. The four cells of the sardines family now come
    # from one generator and one placement procedure, and this is its plain cell, seed
    # 20261169 of `docs/reference/sardines-family-generator.py`. Frozen from
    # `99f3db6e1b59`/`eeab8b10ee1b6543`, itself frozen that morning from
    # `f43d462fe570`/`9ec1b5e290fc928c` when the user gave the pawn its double step.
    # Both older measurements stay in `docs/reference/corpus-replay-2026-09-11.md`,
    # marked superseded.
    "rulesets/sardines.json": ("9f538ab85a77", "685dcd4a16589d0e"),
    "rulesets/shatranj.json": ("67270a8805a0", "14a6ff87ac93cbe0"),
    # Q454 (B3 of `docs/reference/traditional-shogi.md`), 2026-09-25: re-frozen on purpose,
    # the user's decision to make it traditional shogi. Schema 15's `on_leaving` goes on
    # all six promotion entries, so a piece may promote on a move out of its zone too, and
    # perft(5) becomes the published 19,861,490. A different game, not a regression.
    # `runs/630828e31147-*` stay in the main checkout as the old game's measurement.
    # was 630828e31147 / e748341e571a98b9
    "rulesets/shogi.json": ("54c731ca6e95", "ae0af217d07baaa2"),
    "rulesets/threecheck.json": ("b1f536573c22", "72cf8d36fe5262ee"),
    # Q868, 2026-10-12: `rulesets/torus-extended.json` (`fdadd9839d34`/`213502f4d3c8b294`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q869, 2026-10-13: the rear pawn `V` respelled as a pawn placed facing backwards with a
    # facing no move turns (schema 25); the same game move for move (perft and random walks
    # agree), a new hash, `runs/3194d3a2251b-*` kept. was 3194d3a2251b / 83d672ca1a180146
    "rulesets/torus-less-extended.json": ("8d9fb1b4c0b4", "802c13af5c7fbcf1"),
    "rulesets/torus.json": ("5f9c1bfcf85a", "e10795221943c852"),
    # Q434, 2026-09-25: `rulesets/tunnel-horizontal.json` (`195564f716ec`/
    # `d974a81ecff48da0`) was retired here, by the user's decision: its lone promotion
    # `rank: 9` lay past its 8-high board, so its pawns could never promote, and other
    # near-chess wraparound games cover its ground. Its four runs stay in the main
    # checkout (`runs/195564f716ec`), and the validator now refuses such a rank.
    # Q868, 2026-10-12: `rulesets/tunnel-vertical.json` (`0268bb0926bf`/`b74629fce0f2c511`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q313, 2026-09-18: re-frozen from `2437aaa92bb15dba`, and the mechanics hash does
    # not move because no spec file was edited. `compile.py` used to walk a piece's
    # direction *set*, whose tuple hash is 64 bits here and 32 under Emscripten, and a
    # lame leap is appended to the slide table in that order -- so xiangqi's horse and
    # elephant put the interpreter's hash width into `slide_squares` and a browser build
    # played a different game (`docs/reference/browser-core-probe.md`). Sorting the walk
    # makes the tables a property of the spec, and xiangqi is the only ruleset in the
    # tree with a lame leap, so it is the only row that moves.
    "rulesets/xiangqi.json": ("151582aed409", "d3c8d6d4fac6ef02"),
    # Schema 9's own showcase (Q9c) had four rows here -- `boar`, `-bounty`, `-lone`,
    # `-tame`, at `7bf6ee1c0fbc`/`aa00a4c68810c372`, `5b5bfedf821e`/`7210f9a4c95cd6cb`,
    # `22e32e02e45d`/`53b9204937cfcbd2` and `477611bdbb7b`/`cc3addb3bce41ff0`. Q67 removed
    # the specs on 2026-09-07 (`docs/reference/schema-9.md`: the boars die in the opening,
    # so the showcase measured chess with a forced capture), and with them the only
    # rulesets in the tree that exercise a third owner. The pairs above are kept in this
    # comment so a rebuilt showcase can be compared against them.
    #
    # `archive/wolves` (Q68) is the replacement and is the row below rather than a
    # comment: four wild pieces on two mirrored five-square patrols, taken only by a
    # knight, on the board for 0.983 of the plies its campaign played
    # (`docs/reference/wild-piece-that-matters.md`). It is the only spec in the tree that
    # exercises the schema-9 path, so this row is what stops a compiler change to `npcs`
    # from moving a real ruleset's tables in silence. Frozen *at* schema 9 rather than
    # before it -- there is no earlier reading of a rule that did not exist -- which is
    # the same footing the boars' rows had.
    "archive/cannonade.json": ("c2925e027455", "9f90ad036c9a6a53"),
    # Landed beside the bump (Q6b-2, schema 6, no schema-9 key), so this row is frozen
    # from the schema-9 compiler: for a spec without `npcs` every boundary key it adds
    # sits at its inert default and is dropped, so the tables are byte-identical to the
    # schema-8 compiler's by construction -- the same argument the test below makes.
    "archive/hierarchy-extended.json": ("57667d7861ad", "0da72ea2e0260133"),
    # Q868, 2026-10-12: `archive/hierarchy-fall-swap.json` (`f1fdbb539699`/`d1bbe2c7ea75dc5a`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    "archive/hierarchy-fall.json": ("3c639d277808", "c98f538601b4afe5"),
    "archive/hierarchy-rise-stay.json": ("482ddc556331", "353b16ba98f79721"),
    "archive/hierarchy-rise.json": ("1ba3d61e5961", "f5326430599d6230"),
    # Q20, 2026-09-14: schema 12's showcase, `hierarchy-rise` under `position: "choose"`.
    # The only spec in the tree that compiles a landing table, so this row pins that path.
    "archive/hierarchy-rise-throw.json": ("78130df01edd", "5f32c46b0cd9e3f9"),
    # Q21, 2026-09-14: schema 13's showcase, chess with sweeping bishops. The only spec in
    # the tree that compiles a sweep table, so this row pins that path.
    "archive/chess-sweeping-bishops.json": ("42bd7d06828d", "de55d69b00587dc2"),
    "archive/marchfield.json": ("0be72e61ef24", "3a2d5a7fddb76113"),
    # Schema 10's seven showcases (Q101, 2026-09-07), rows 23--30 of
    # `docs/reference/schema-showcase-coverage.md`. Frozen *at* schema 10 rather than
    # before it, on `wolves`' footing: there is no earlier reading of a rule that did not
    # exist. What these rows are for is the same thing `wolves`' row is for -- they are
    # the only specs in the tree that exercise a contact table, a narrowed explosion, a
    # stock or a never-repeating clock, so a compiler change to any of the five would
    # otherwise move no committed ruleset's tables and pass in silence.
    "archive/merger.json": ("85555467a4ca", "d4079562891aafd2"),
    "archive/perpetual.json": ("df2fbba0ca76", "3f3573664d563043"),
    # Q880, 2026-10-13: a one-square diagonal step passes over nothing under
    # `blocks_leaps` (`compile._lame_path` halves toward zero, not floor), so a wall
    # beside a king or a pawn no longer stops two of its four diagonals. Quarry is the
    # only spec in the tree that moved; re-measured, the old score kept under `previous/`.
    "archive/quarry.json": ("51d9b36b4e93", "7bed2a6503798054"),  # was b03c53fbdc1c2346
    "archive/queen-bomb.json": ("313c913a5b3b", "fa021e91cdc7782d"),
    "archive/queens-wake.json": ("328cab4db5cc", "185b194e58d513eb"),
    "archive/reckoning.json": ("ffc7b604bea0", "165fff189589d44f"),
    # The two stone games moved **both** numbers on 2026-09-08 (Q131), which is the row
    # kind the header calls out: a ruleset whose *spec* is edited, not a compiler
    # regression. Both descriptions have always said a stone "is never captured and never
    # leaves", neither spec carried a capture matrix, and the user's call was that the
    # description is the specification -- so each gained six `captures` lists and each is
    # a different game. The fingerprint had to move on its own account: forbidding the
    # capture of a stone removes moves from the move list, which is why this is *not* the
    # `wolves-immune` case of a re-spelling that compiles to the same tables. Frozen from
    # `d9a411e36af4`/`3bf67469f22c8725` and `707d1f6f1dbc`/`23b9fb4a264a179b`, the rows
    # the superseded specs held after Q113 -- and the runs kept under those two hashes
    # measure the specs, not these rows.
    "archive/stones.json": ("93b02b591c87", "8b2add48413f7969"),
    "archive/succession.json": ("e7293070b5be", "ce754901363a2c3b"),
    "archive/three-stones.json": ("03aa7c3f2f18", "b3d9e4ca986272a9"),
    # The sardines family's wrap-and-facing cell, Q180, 2026-09-11, seed 20262067. It
    # replaces `weathervane-chaos-torus.json` (`a6ebba755d77`/`26fdee19415a4509`), which
    # the user landed in `95e1495` that morning from a generator of its own.
    # Q868, 2026-10-12: `archive/weathervane-torus.json` (`9777cd6c9ed1`/`910ad7686ab0c31d`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q28b, 2026-09-13: `weathervane` with a two-axis pawn whose `initial` is backward,
    # rows 15 and 16 of `docs/reference/schema-showcase-coverage.md`. The only spec in the
    # tree that compiles a two-axis piece, so this row pins that path.
    # Q869, 2026-10-13: the attack sources keep only the frames a type can stand in, so a
    # piece that faces no way loses its sources for the facings it never holds; the same
    # games move for move (played and perft-compared both ways). was 0664986abe7f9b25
    "archive/weathervane-half-turn.json": ("8fc3939243d5", "7e9a6e3b57386450"),
    # The family's facing cell, Q180, seed 20261254. Re-frozen from
    # `b88e98dab286`/`55501435ca71e297`, which was the pre-edit sardines plus facing and
    # so had stopped being the plain cell's control. This one is `rulesets/sardines`'
    # army and rules with `facing` on the pawn, by construction rather than by a test
    # noticing afterwards, which is what `test_only_the_pawn_differs_from_sardines` reads.
    # Q869, 2026-10-13: the attack sources keep only the frames a type can stand in, so a
    # piece that faces no way loses its sources for the facings it never holds; the same
    # games move for move (played and perft-compared both ways). was 462cb3cb5476254f
    "archive/weathervane.json": ("1442513d38d7", "ba2232e42e5b3915"),
    "archive/wolves.json": ("ceb4916bdabc", "5a0c6e8ef7f53176"),
    # `wolves-immune` is `wolves` re-spelled with `npcs[].immune` and its tables are
    # `wolves`' but for the Zobrist draws, which the fingerprint covers -- so the two
    # rows differ, and that difference is the mechanics hash reseeding the draw rather
    # than any rule compiling differently.
    # Q876, 2026-10-13: `archive/wolves-immune.json` (`a2193fb37097`/`5f1cc21fe83d08c8`) retired by the library audit's second pass for `archive/pitchfork.json` (`docs/archive/plan-library-audit.md`, Q876); its runs stay.
    # Q76, 2026-09-13: `wolves` plus `scoring.count: "points"` and a bounty of 1, rows 19
    # and 22 of `docs/reference/schema-showcase-coverage.md`. The only spec in the tree
    # that compiles a non-default bounty and a points count, so this row pins that path.
    "archive/wolves-bounty.json": ("54b38fb8cba8", "6435d2c15d411b4d"),
    # Q77, 2026-09-14: `wolves` with both routes rotated by one square.
    # Q868, 2026-10-12: `archive/wolves-rotated.json` (`213f7b6604e4`/`0db063d05e21dd56`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q222, 2026-09-14: the other three rotations, by two, three and four squares.
    # Q868, 2026-10-12: `archive/wolves-rotated-2.json` (`8ec0ce4678e9`/`112bb4d97b35232f`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q868, 2026-10-12: `archive/wolves-rotated-3.json` (`2e94e1688601`/`e71396ec37a17a5f`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q868, 2026-10-12: `archive/wolves-rotated-4.json` (`87a488975531`/`7863f23a3bbd4b4e`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q229, 2026-09-15: schema 14's showcase, Glinski-style hexagonal chess on an axial
    # rhombus with thirty walls, `lattice: "hex"` declared and a promotion zone on the
    # hexagon's far edge. The only spec in the tree that compiles a zone that is not whole
    # ranks, so this row pins that path. `docs/reference/hex-convention.md`.
    # Q280, 2026-09-17: the square-grid parent `archive/hex-glinski.json`
    # (`5228694f93bd`) was retired here; schema 14 made its convention a declaration.
    # Q868, 2026-10-12: `archive/hex-glinski-promoting.json` (`279f92a97530`/`b429acf0d4a10f89`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q281a4, 2026-09-28: schema 16's showcases. `hex-glinski-hexagon` is the parent above
    # with `board.shape: "hexagon"` for its thirty walls. Its tables are the parent's
    # array by array but for the five Zobrist arrays, drawn from the mechanics hash as
    # `wolves-immune`'s are, so its fingerprint differs and that is the only reason
    # (`tests/test_schema_sixteen.py::TestTheTwin`). `flanks` joins its empty flanks
    # (`nw_se: "both"`), `flanks-one-way` the same seam one way per player.
    "archive/hex-glinski-hexagon.json": ("6eaaad41673b", "502b6c432a0489b4"),
    # Q870, 2026-10-13: `archive/hex-glinski-flanks.json` (`5ba53615c752`/`e615ab9b05ef7ce9`) retired by the library audit's second pass for `archive/pinwheel.json` (`docs/archive/plan-library-audit.md`, Q870); its runs stay.
    # Q870, 2026-10-13: `archive/hex-glinski-flanks-one-way.json` (`2edcd9a30022`/`4b8332fedb7ee7a7`) retired by the library audit's second pass for `archive/pinwheel.json` (`docs/archive/plan-library-audit.md`, Q870); its runs stay.
    # Q540, 2026-10-09: schema 17's showcases, chess with one rule each about what a
    # capture leaves behind (rifle, circe, Andernach, kamikaze).
    "archive/rifle.json": ("f7a789809758", "0fb33501ce9f14b3"),
    "archive/circe.json": ("54a0a193da0f", "33948d3b415bd51e"),
    "archive/andernach.json": ("f9e5dbf3ff1f", "05eec37015be2352"),
    "archive/kamikaze.json": ("6cd3785c7246", "e626905b911bbd2b"),
    # Q787, 2026-10-09: the three showcases the 2026-10-09 audit found missing (gaps G1 to G3
    # of `docs/reference/schema-showcase-coverage.md`): shogi plus `promote_on_entry`, shogi
    # with a backward-facing dropped pawn, and chess with a pawn that knocks pieces down.
    "archive/shogi-drop-promoted.json": ("ffb9a99dbd01", "6811d5a31d92a207"),
    # Q869, 2026-10-13: the attack sources keep only the frames a type can stand in, so a
    # piece that faces no way loses its sources for the facings it never holds; the same
    # games move for move (played and perft-compared both ways). was b70216a4c181360f
    "archive/shogi-backward-drops.json": ("8478a3301cfd", "fd7946431f88a7cb"),
    "archive/pawn-knockdown.json": ("d50c552b49da", "89578d839100f40d"),
    # Q781, 2026-10-09: schema 18a's showcases, Q533's audit specs with the rule each could
    # not say: Einstein chess's keeping link and Omega chess's castling on a walled rank.
    "archive/einstein.json": ("d90706fec9c0", "996d6281b00fcd83"),
    "archive/omega.json": ("c68b69c57219", "59fe2b96b67ee042"),
    # Q783, 2026-10-09: schema 18c's showcases, Q533's audit specs with the rule each could
    # not say: Extinction chess's count per type and Grand chess's promotion from the lost.
    "archive/extinction-chess.json": ("ca1fdd0b00bb", "5fdfec30333e4dba"),
    "archive/grand.json": ("393dfeada70d", "dfd7945f0fab0e70"),
    # Q784, 2026-10-09: schema 18d's showcases for the placement phase: Placement chess
    # (less its opposite-coloured bishops and its dropped castling) and Sittuyin (with a
    # caveat: a pawn promotes on arrival, not in place).
    "archive/placement-chess.json": ("2cb9cb3c018c", "230ba2d2ea235f42"),
    "archive/sittuyin.json": ("771c1a22e182", "59fcaa6f1987dac3"),
    # Q785, 2026-10-09: schema 18e's showcase for dropping either face: Kyoto shogi, Q533's
    # audit spec with each turned face's promoted_from and the flag.
    "archive/kyoto.json": ("a1805ac1adc1", "e18bfb3c15b95c45"),
    # Q786, 2026-10-09: schema 18f's showcase for winning by leaving a square: Jeson Mor,
    # Q533's audit spec (jeson-mor-approx, 9447c056a0c0) with reach.on "leave".
    "archive/jeson-mor.json": ("fb8a67625131", "afef964377992171"),
    # Q782, 2026-10-10: schema 18b's showcases, `hex-glinski-promoting` with every officer's
    # move written once and turned by the hexagon's `rotate6`, and by `dihedral12`. Their
    # tables are the parent's array by array but for the five Zobrist draws, which the
    # mechanics hash seeds (`tests/test_schema_eighteen.py::TestHexSymmetries`).
    # Q870, 2026-10-13: `archive/hex-glinski-turned.json` (`30c1003b9a61`/`88ab61fbcf782f93`) retired by the library audit's second pass for `archive/pinwheel.json` (`docs/archive/plan-library-audit.md`, Q870); its runs stay.
    # Q870, 2026-10-13: `archive/hex-glinski-turned-mirrored.json` (`461e29677d6d`/`af0f34c7a089f751`) retired by the library audit's second pass for `archive/pinwheel.json` (`docs/archive/plan-library-audit.md`, Q870); its runs stay.
    # Q843a, 2026-10-11: the eight portal rows' fingerprints were re-frozen (the old one on
    # each row's comment). Schema 21 renamed and re-encoded two boundary keys,
    # `portal_partner` -> `portal_dest` and `portal_open` (a bool per player) ->
    # `portal_admits` (a piece-type mask per player), and added `portal_group`; every other
    # array of the eight is unchanged, `dest` equals the old `partner` and `admits` is
    # non-zero exactly where `open` was true (compared against the schema 20 core), and every
    # perft of the eight is unchanged (`tests/test_schema_twenty_one.py::FROZEN_PERFT`). No
    # mechanics hash moved (`docs/archive/plan-portals-2.md` §2.4).
    # Q772a, 2026-10-10: schema 19's showcases (`docs/archive/plan-portals.md` §7 P4): the three games
    # Divide (capablanca on a walled 10x11), Ferry (losalamos, two pairs either player
    # moves) and Carousel (losalamos on a 6x8 cylinder, a private lift each), and the four
    # one-feature specs: Wormhole on gardner, its `moved_by` and `passage` arms, and a
    # two-exit pair on cylinder (`docs/reference/portal-showcases.md`).
    "archive/divide.json": ("1f85f50a23f6", "b2afc37a1693de09"),  # was d0ab03b90fdc996f
    "archive/ferry.json": ("1501615e08e2", "e7770d3fc50aee5a"),  # was 6a3e2afa90281e2e
    "archive/carousel.json": ("f1e40e35a6d9", "69a08feee1d2d3ae"),  # was 77a125f71060e8d6
    "archive/wormhole.json": ("3732e55621b1", "f19895f2038e8832"),  # was b0b3d89367aea3fe
    # Q868, 2026-10-12: `archive/wormhole-adrift.json` (`22cc8982bc90`/`2929f116a0b775bb`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    "archive/shut-wormhole.json": ("58afebca2286", "70627853a1fe6053"),  # was a4db7f4ed8334ea1
    "archive/cylinder-wormhole.json": ("5abb0b804cf6", "f4527ef755dee62c"),  # was 44ff210388e3b982
    # Q807, 2026-10-10: schema 20's showcase, Ferry with `no_straight_back` on both pairs
    # (`docs/archive/plan-portals.md` *Ferry's shuffle*), a new game beside Ferry, whose row is unmoved.
    # was 1fdb2020deedb6d1
    "archive/ferry-no-straight-back.json": ("dd17d71412a7", "0bdeb43fbc2e4d7d"),
    # Q850, 2026-10-11: schema 23's showcase, the user's game -- two wolves on one
    # sixteen-square circuit, `also_at` (`designer-case-study-wolves.md` §2.2). New.
    "archive/wolves-patrol.json": ("b2011455c4af", "5b144080a51061e1"),
    # Q772e, 2026-10-12: Divide's option B, the two-way door (`docs/archive/plan-portals.md` *Divide's
    # return rule*): `divide` with the pair's `passage` removed, a new game beside it. New.
    # Q868, 2026-10-12: `archive/divide-two-way.json` (`2fa712feb218`/`61e519fb59ea4574`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q772f, 2026-10-12: Divide's option C, both ends fixed (`docs/archive/plan-portals.md` *Divide's
    # return rule*): `divide-two-way` with the pair's `moved_by` removed, a new game. New.
    # Q868, 2026-10-12: `archive/divide-fixed.json` (`31ead85da009`/`f5bf221dce8ccfe8`) retired by the library audit (`docs/archive/plan-library-audit.md`); its runs stay.
    # Q870, 2026-10-13: the hexagon set's designed game, built around what four retired
    # variants of `hex-glinski-hexagon` were kept to show: a one-handed knight written
    # `rotate6` beside `dihedral12` officers, and a one-way seam on a 61-cell hexagon
    # (`docs/archive/plan-library-audit.md`; `tests/test_pinwheel.py`). New.
    "archive/pinwheel.json": ("61c4e247dc3c", "9cc78fa02cf12867"),
    # Q876, 2026-10-13: the game built around `npcs[].immune` in place of `wolves-immune`:
    # eight wolves in pairs (`also_at`) on the centre rings, and only a pawn may take one
    # (`docs/archive/plan-library-audit.md`; `tests/test_pitchfork.py`). New.
    "archive/pitchfork.json": ("d2a722b461ff", "f7fe75d2793c461a"),
    # Q844b, 2026-10-13: the portal showcases on the rules as they now stand
    # (`docs/reference/portal-showcase-ideas.md`; `tests/test_portal_showcases_q844b.py`):
    # Sluice, chess with a momentum funnel a side (schema 21's maps), Crease, Capablanca
    # with two fold pairs (schema 24), and pawnproof wormhole, `wormhole` with `carries`.
    # New (Q844's drafts of 2026-10-11 were never committed to main).
    "archive/sluice.json": ("f3c624d2f297", "f84bca5f42fba51f"),
    "archive/crease.json": ("0448722861a3", "ac98338606812bdf"),
    "archive/pawnproof-wormhole.json": ("feaad3754a7e", "fa4f817969d2b144"),
    # Q918e, 2026-10-15: the showcases of schema 26's simultaneous moves
    # (`docs/archive/plan-simultaneous.md`, *Q918e*; `tests/test_showcases_q918e.py`): Gardner in
    # rounds (the mechanics of `tests/_simultaneous_gardner.json`), with meek pawns
    # (`beats`), with the initiative, and the watering hole (ties impossible). New.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds.json` (`e043fce6968e`/`8a0183446ffc68f9`) retired from the previews
    # (`docs/reference/previews-2026-10-18.md`), its file kept as `tests/_retired_*.json`; its runs stay.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-with-meek-pawns.json` (`ff1ccdaaf3cf`/`95e105b3ac964b51`) retired from the previews
    # (`docs/reference/previews-2026-10-18.md`), its file kept as `tests/_retired_*.json`; its runs stay.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-with-the-initiative.json` (`6278607c721b`/`8c4ede64bdaf6f6e`) retired from the previews
    # (`docs/reference/previews-2026-10-18.md`), its file kept as `tests/_retired_*.json`; its runs stay.
    "archive/watering-hole.json": ("5c0b017473f5", "f264b4cdd576ed0b"),
    # Q942c, 2026-10-17: the showcases of schema 27's three switches against the royal chase
    # (`docs/archive/plan-round-chase.md` §5; `tests/test_showcases_q942.py`): Gardner in rounds
    # resting (`rest`, the mechanics of `tests/_simultaneous_resting.json`), where swaps
    # collide (`swap`), scored (`both`), and against the chase (all three, the mechanics of
    # `tests/_simultaneous_scored.json`). New.
    "archive/gardner-in-rounds-resting.json": ("4da669bb3d50", "f75ed40edaaf4658"),
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-where-swaps-collide.json` (`7985900696d7`/`84d7ffe03c147c54`) retired from the previews
    # (`docs/reference/previews-2026-10-18.md`), its file kept as `tests/_retired_*.json`; its runs stay.
    # Q962b, 2026-10-18: `archive/gardner-in-rounds-scored.json` (`a90a8db7e2dd`/`66e201715549ce34`) retired from the previews
    # (`docs/reference/previews-2026-10-18.md`), its file kept as `tests/_retired_*.json`; its runs stay.
    # Q962, 2026-10-18: against the chase renamed `archive/breather.json`, name and description only;
    # the mechanics hash is the same.
    "archive/breather.json": ("632c199e4589", "359f27e3538d5415"),
    # Q935f, 2026-10-17: the spy's showcases (`docs/plan-spy.md` §5; `tests/test_showcases_q935f.py`):
    # Gardner with spies (SP1, `pieces.spy`, the mechanics of Q935d's draft) and Gardner with
    # spies resting (SP2, with `rest`). New.
    # Q962b, 2026-10-18: `archive/gardner-with-spies.json` (`f4c62a33798d`/`7e3a92c5d12b44dd`) retired from the previews
    # (`docs/reference/previews-2026-10-18.md`), its file kept as `tests/_retired_*.json`; its runs stay.
    "archive/gardner-with-spies-resting.json": ("1d798bedbf66", "8a309ac4ab581f8e"),
    # Q936s, 2026-10-17: the showcase of `"per_type": "own"`, two armies of different kinds
    # (`docs/reference/extinction-any-kind.md` §5; `tests/test_showcases_q936s.py`). New;
    # measured first as `30c055a056ec` (tables `d53eb61a71b4745e`), re-frozen the same day when
    # the champion's and elephant's moves were split one descriptor per step length, the same
    # game move for move (§5.4).
    "archive/riders-and-leapers.json": ("26636602fae0", "8e384b6a8a7868c2"),
}


def _boar(chess_spec: dict[str, Any], **rules: Any) -> dict[str, Any]:
    """The stage-C showcase, as the plan pre-registers it: chess plus a wild boar.

    Two boars, one on each side's third and fourth ranks, because the centre loop's
    mirror image is itself and the validator refuses that. ``W`` because chess has a
    bishop on ``B``. The boar's own moves are a knight's -- what it does in a player's
    hands, never how it walks.
    """
    spec = copy.deepcopy(chess_spec)
    spec["schema_version"] = 9
    spec["name"] = "boar"
    spec["pieces"].append(
        {
            "id": "W",
            "name": "boar",
            "value": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
        }
    )
    spec["rules"]["npcs"] = [
        {"piece": "W", "route": ["c3", "c4", "d4", "d3"], "bounty": 3, "mirror": True}
    ]
    spec["rules"]["scoring"] = {"count": "points"}
    spec["rules"].update(rules)
    return spec


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


# ------------------------------------------------------------------ the bump moved nothing


class TestTheBumpMovedNothing:
    @pytest.mark.parametrize("path", sorted(FROZEN), ids=lambda p: Path(p).stem)
    def test_every_ruleset_compiles_to_the_tables_it_did_before_schema_nine(
        self, path: str
    ) -> None:
        """Byte-identical canonical tables, pinned rather than claimed.

        The fingerprint is over `bridge.canonical_tables`, with every boundary key at its
        inert default dropped -- so `has_npcs: False` and two empty tables move nothing,
        which is exactly what `_INERT_DEFAULTS` exists to make true.
        """
        spec = load_spec(ROOT / path)
        digest, tables = FROZEN[path]
        assert spec_hash(spec) == digest, f"{path} is not the file that was frozen"
        assert tables_fingerprint(spec) == tables, f"{path} compiles to different tables"

    def test_the_frozen_table_covers_the_whole_corpus(self) -> None:
        """A ruleset added later gets a named failure here, not a silent gap."""
        on_disk = {
            f"{folder}/{path.name}"
            for folder in ("rulesets", "archive")
            for path in (ROOT / folder).glob("*.json")
        }
        assert on_disk == set(FROZEN), sorted(on_disk ^ set(FROZEN))

    def test_a_ruleset_without_wild_pieces_is_the_shape_it_always_was(
        self, chess_spec: dict[str, Any]
    ) -> None:
        compiled = compile_spec(chess_spec)
        assert compiled.npcs is None
        assert compiled.n_owners == 2
        assert len(compiled.attack_sources) == 2
        assert len(compiled.zobrist_pieces) == 2 * compiled.n_facings * compiled.n_types + 1
        assert rust_supports(compiled) is None
        tables = canonical_tables(compiled)
        assert not {"has_npcs", "npc_next", "npc_bounty"} & set(tables)


# ---------------------------------------------------------------- what the validator says


class TestWhatTheValidatorRefuses:
    def test_the_showcase_validates(self, chess_spec: dict[str, Any]) -> None:
        assert validate_spec(_boar(chess_spec)) == []

    def test_a_wild_royal_is_refused(self, chess_spec: dict[str, Any]) -> None:
        spec = _boar(chess_spec)
        spec["rules"]["npcs"][0]["piece"] = "K"
        assert any("royal" in error for error in validate_spec(spec))

    def test_a_route_may_not_visit_a_square_twice(self, chess_spec: dict[str, Any]) -> None:
        spec = _boar(chess_spec)
        spec["rules"]["npcs"][0]["route"] = ["c3", "c4", "c3"]
        assert any("twice" in error for error in validate_spec(spec))

    def test_two_routes_may_not_share_a_square(self, chess_spec: dict[str, Any]) -> None:
        spec = _boar(chess_spec)
        spec["rules"]["npcs"].append({"piece": "W", "route": ["e3", "d4"]})
        errors = validate_spec(spec)
        assert any("both walk d4" in error for error in errors), errors

    def test_the_centre_loop_is_its_own_mirror_image_and_is_refused(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """The first thing a designer will meet, and the plan says so."""
        spec = _boar(chess_spec)
        spec["rules"]["npcs"][0]["route"] = ["d4", "d5", "e5", "e4"]
        errors = validate_spec(spec)
        assert any("mirrored" in error and "both walk" in error for error in errors), errors

    def test_a_wild_piece_may_not_start_under_the_opening_array(
        self, chess_spec: dict[str, Any]
    ) -> None:
        spec = _boar(chess_spec)
        spec["rules"]["npcs"][0]["route"] = ["a2", "a3"]
        errors = validate_spec(spec)
        assert any("starts on a2" in error for error in errors), errors
        # The mirrored start too: a3's image is a6, and a6 is free, but a7 is not.
        spec["rules"]["npcs"][0]["route"] = ["a7", "a6"]
        spec["rules"]["npcs"][0]["mirror"] = False
        assert any("starts on a7" in error for error in validate_spec(spec))

    def test_a_route_may_not_cross_a_wall(self, chess_spec: dict[str, Any]) -> None:
        """The sixth rule that names a square nobody can stand on."""
        spec = _boar(chess_spec, obstacles={"squares": ["c4"]})
        errors = validate_spec(spec)
        assert any("walks through an obstacle at c4" in error for error in errors), errors

    @pytest.mark.parametrize(
        ("rule", "value", "word"),
        [
            (
                "strike",
                {"effect": "demote", "position": "stay", "immune": ["K"],
                 "chain": [{"from": "Q", "to": "R"}]},
                "strike",
            ),
            ("explode_on_capture", {"radius": 1}, "explode"),
            ("turn", {"moves": 2, "ramp": [1]}, "turn"),
        ],
    )
    def test_three_combinations_are_refused_by_name(
        self, chess_spec: dict[str, Any], rule: str, value: Any, word: str
    ) -> None:
        spec = _boar(chess_spec, **{rule: value})
        errors = validate_spec(spec)
        assert any("rules.npcs" in error and word in error for error in errors), errors

    def test_a_strike_that_compiles_to_nothing_is_not_a_strike(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """`{"effect": "capture"}` is plain capture and refuses nothing."""
        spec = _boar(chess_spec, strike={"effect": "capture"})
        assert validate_spec(spec) == []

    def test_drops_are_not_refused(self, chess_spec: dict[str, Any]) -> None:
        """A captured boar in hand and back on the board as your own is a design."""
        spec = _boar(chess_spec, drops={"from_captures": True})
        assert validate_spec(spec) == []

    def test_points_need_something_that_pays(self, chess_spec: dict[str, Any]) -> None:
        spec = _boar(chess_spec)
        del spec["rules"]["npcs"]
        assert any("nothing to count" in error for error in validate_spec(spec))
        spec = _boar(chess_spec)
        spec["rules"]["npcs"][0]["bounty"] = 0
        assert any("bounty above zero" in error for error in validate_spec(spec))

    def test_a_bounty_defaults_to_the_value_and_pays(self, chess_spec: dict[str, Any]) -> None:
        spec = _boar(chess_spec)
        del spec["rules"]["npcs"][0]["bounty"]
        assert validate_spec(spec) == []

    @pytest.mark.parametrize(
        ("edit", "word"),
        [
            (lambda npc: npc.__setitem__("piece", "X"), "not a declared piece"),
            (lambda npc: npc.__setitem__("bounty", -1), "0 or more"),
            (lambda npc: npc.__setitem__("bounty", True), "0 or more"),
            (lambda npc: npc.__setitem__("mirror", "yes"), "boolean"),
            (lambda npc: npc.__setitem__("behaviour", "patrol"), "unknown keys"),
            (lambda npc: npc.__setitem__("route", []), "non-empty"),
            (lambda npc: npc.__setitem__("route", ["z9"]), "not a square"),
        ],
    )
    def test_a_malformed_entry_is_refused_by_name(
        self, chess_spec: dict[str, Any], edit: Any, word: str
    ) -> None:
        spec = _boar(chess_spec)
        edit(spec["rules"]["npcs"][0])
        errors = validate_spec(spec)
        assert any(word in error for error in errors), errors

    def test_an_empty_list_is_refused(self, chess_spec: dict[str, Any]) -> None:
        spec = _boar(chess_spec)
        spec["rules"]["npcs"] = []
        assert any("non-empty" in error for error in validate_spec(spec))

    def test_wild_pieces_declare_schema_nine(self, chess_spec: dict[str, Any]) -> None:
        """Both schema-9 features, in one sentence, so a spec is told everything at once."""
        spec = _boar(chess_spec)
        spec["schema_version"] = 8
        errors = validate_spec(spec)
        assert len(errors) == 1, errors
        assert "schema 9" in errors[0] and "rules.npcs" in errors[0] and "points" in errors[0]
        # And `points` alone, with no wild piece, is still a schema-9 word.
        plain = copy.deepcopy(chess_spec)
        plain["schema_version"] = 8
        plain["rules"]["scoring"] = {"count": "points"}
        assert any("schema 9" in error for error in validate_spec(plain))

    def test_three_owners_share_the_one_byte(self, chess_spec: dict[str, Any]) -> None:
        """Three owners at four facings is 21 piece types, not 24."""
        spec = _boar(chess_spec)
        spec["pieces"][0]["facing"] = {"axes": 4}  # the pawn
        while len(spec["pieces"]) < 22:
            n = len(spec["pieces"])
            spec["pieces"].append(
                {"id": f"{n:X}"[-2:] if n > 9 else str(n), "value": 1,
                 "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]}
            )
        errors = validate_spec(spec)
        assert any("one byte" in error and "wild" in error for error in errors), errors
        del spec["rules"]["npcs"]
        del spec["rules"]["scoring"]
        assert not any("one byte" in error for error in validate_spec(spec))


# ------------------------------------------------------------------- what the compiler builds


class TestWhatTheCompilerBuilds:
    def test_a_wild_piece_is_a_third_owner_on_the_opening_board(
        self, chess_spec: dict[str, Any]
    ) -> None:
        compiled = compile_spec(_boar(chess_spec))
        boar = compiled.piece_ids.index("W")
        assert compiled.n_owners == 3
        for start in ("c3", "c6"):
            value = compiled.initial_board[_square(start)]
            assert value == compiled.encode_piece(WILD, boar)
            assert compiled.owner_of(value) == WILD
            assert compiled.type_of(value) == boar
            assert compiled.facing_of(value) == 0
            # No unmoved bit: a wild square is neither a castling right nor a first move.
            assert not compiled.initial_unmoved >> _square(start) & 1
        # Above both armies, contiguously: the player-major property with a third block.
        assert compiled.encode_piece(WILD, 0) == 2 * compiled.side_stride + 1

    def test_the_route_is_one_table_and_it_cycles(self, chess_spec: dict[str, Any]) -> None:
        compiled = compile_spec(_boar(chess_spec))
        npcs = compiled.npcs
        assert npcs is not None
        for route in (["c3", "c4", "d4", "d3"], ["c6", "c5", "d5", "d6"]):
            squares = [_square(name) for name in route]
            for here, there in zip(squares, squares[1:] + squares[:1], strict=True):
                assert npcs.next[here] == there
                assert npcs.bounty[here] == 3.0
                assert npcs.piece[here] == compiled.piece_ids.index("W")
        off = _square("e4")
        assert (npcs.next[off], npcs.bounty[off], npcs.piece[off]) == (-1, 0.0, -1)
        assert npcs.starts == ((_square("c3"), 6), (_square("c6"), 6))

    def test_a_mirrored_route_is_walked_in_the_same_order_reflected(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """Under `orientation: rotate` the image flips the file too."""
        spec = _boar(chess_spec)
        spec["setup"]["orientation"] = "rotate"
        npcs = compile_spec(spec).npcs
        assert npcs is not None
        image = [_square(name) for name in ("f6", "f5", "e5", "e6")]
        for here, there in zip(image, image[1:] + image[:1], strict=True):
            assert npcs.next[here] == there

    def test_the_bounty_defaults_to_the_declared_value(self, chess_spec: dict[str, Any]) -> None:
        spec = _boar(chess_spec)
        del spec["rules"]["npcs"][0]["bounty"]
        spec["pieces"][-1]["value"] = 2.5
        npcs = compile_spec(spec).npcs
        assert npcs is not None
        assert npcs.bounty[_square("c3")] == 2.5

    def test_a_one_square_route_is_a_piece_that_never_moves(
        self, chess_spec: dict[str, Any]
    ) -> None:
        spec = _boar(chess_spec)
        spec["rules"]["npcs"][0]["route"] = ["c3"]
        compiled = compile_spec(spec)
        assert compiled.npcs is not None
        assert compiled.npcs.next[_square("c3")] == _square("c3")
        # ...and threatens nothing: no attack source anywhere in the third block.
        assert all(not sources for sources in compiled.attack_sources[WILD])

    def test_its_threat_is_the_square_it_steps_to_next(self, chess_spec: dict[str, Any]) -> None:
        """The third owner's attack block, built from the routes rather than the tables."""
        compiled = compile_spec(_boar(chess_spec))
        assert len(compiled.attack_sources) == 3
        boar = compiled.encode_piece(WILD, compiled.piece_ids.index("W"))
        sources = compiled.attack_sources[WILD][_square("c4")]
        assert sources == ((_square("c3"), boar, (), (), 0, False, 0, 0, ()),)
        # Every route square but the target of nothing is attacked from exactly one square.
        attacked = {square for square in range(64) if compiled.attack_sources[WILD][square]}
        assert attacked == {_square(n) for n in ("c3", "c4", "d4", "d3", "c6", "c5", "d5", "d6")}
        # And nothing off the routes is.
        assert not compiled.attack_sources[WILD][_square("e4")]

    def test_the_zobrist_table_has_a_row_per_wild_value(self, chess_spec: dict[str, Any]) -> None:
        compiled = compile_spec(_boar(chess_spec))
        assert len(compiled.zobrist_pieces) == 3 * compiled.n_facings * compiled.n_types + 1

    def test_the_boundary_carries_the_two_tables_and_the_flag(
        self, chess_spec: dict[str, Any]
    ) -> None:
        compiled = compile_spec(_boar(chess_spec))
        tables = canonical_tables(compiled)
        assert tables["has_npcs"] is True
        assert len(tables["npc_next"]) == len(tables["npc_bounty"]) == 64
        assert tables["npc_next"][_square("d3")] == _square("c3")
        assert tables["npc_bounty"][_square("d3")] == 3.0
        assert tables["scoring_count"] == 4
        # The third attack block travels in the same CSR arrays, one more block long.
        assert len(tables["attack_index"]) == 3 * 64 + 1

    def test_the_rule_moves_the_fingerprint_and_the_type_alone_does_not_move_it_back(
        self, chess_spec: dict[str, Any]
    ) -> None:
        with_boars = _boar(chess_spec)
        type_only = copy.deepcopy(with_boars)
        del type_only["rules"]["npcs"]
        del type_only["rules"]["scoring"]
        assert tables_fingerprint(with_boars) != tables_fingerprint(type_only)


# ------------------------------------------------------------------------------ the guard


class TestTheGuardCameOff:
    """Q9b taught the core the rule and lifted the ninth hold.

    The refusal is kept as a *test* rather than as a constant nobody reads: the flag is
    what said the compiler and the core were in step, and turning it back off has to
    still produce the refusal by name, or the next feature's guard is a comment.
    """

    def test_the_core_plays_a_wild_ruleset(self, chess_spec: dict[str, Any]) -> None:
        compiled = compile_spec(_boar(chess_spec))
        assert rust_supports(compiled) is None
        engine = Engine(_boar(chess_spec))
        board = engine.initial_state().board
        wild = [square for square, value in enumerate(board)
                if value and compiled.owner_of(value) == WILD]
        assert wild == sorted(_square(name) for name in ("c3", "c6"))

    def test_the_flag_off_still_refuses_by_name(
        self, chess_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bridge, "_CORE_KNOWS_NPCS", False)
        reason = rust_supports(compile_spec(_boar(chess_spec)))
        assert reason is not None and "rules.npcs" in reason
        with pytest.raises(NotImplementedError, match="rules.npcs"):
            Engine(_boar(chess_spec))

    def test_the_smoke_gate_plays_the_candidate(self, chess_spec: dict[str, Any]) -> None:
        from chess_variant_sandbox.loop.gates import smoke_test

        result = smoke_test(_boar(chess_spec), games=4)
        assert not any("core cannot play" in reason for reason in result.reasons)

    def test_the_fuzzer_walks_it(
        self, chess_spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        boar = _boar(chess_spec)
        monkeypatch.setattr(
            fuzz, "case",
            lambda seed, corpus=fuzz.CORPUS: fuzz.Case(seed, "boar", (), copy.deepcopy(boar)),
        )
        coverage = fuzz.run(range(3), plies=2)
        assert coverage.walked == 3
        assert not coverage.refused
        assert coverage.singles["npcs"] == 3


# --------------------------------------------------------------------------- the surfaces


class TestTheSurfaces:
    def test_the_model_can_propose_it(self) -> None:
        from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
        from chess_variant_sandbox.spec.schema import _NPC_KEYS

        rules = rule_spec_schema()["properties"]["rules"]["properties"]
        assert set(rules["npcs"]["items"]["properties"]) == _NPC_KEYS
        assert rules["npcs"]["items"]["required"] == ["piece", "route"]
        assert "points" in rules["scoring"]["properties"]["count"]["enum"]

    def test_the_prose_says_what_a_wild_piece_does(self, chess_spec: dict[str, Any]) -> None:
        from chess_variant_sandbox.spec.describe import describe_spec

        spec = _boar(chess_spec)
        prose = describe_spec(spec)
        assert "wild boar" in prose and "c3" in prose and "eats any piece" in prose
        assert "in check from it" in prose and "worth 3 points" in prose
        assert "mirror image" in prose and "taking wild pieces" in prose
        without = copy.deepcopy(spec)
        del without["rules"]["npcs"]
        del without["rules"]["scoring"]
        assert describe_spec(without) != prose
        # A moving wall is said to be one.
        tame = copy.deepcopy(spec)
        tame["pieces"][-1]["captures"] = []
        assert "eats nothing and simply blocks" in describe_spec(tame)
        # A post is said to be one.
        post = copy.deepcopy(spec)
        post["rules"]["npcs"][0]["route"] = ["c3"]
        assert "never moves" in describe_spec(post)

    def test_the_rule_is_charged_beside_territory(self, chess_spec: dict[str, Any]) -> None:
        from chess_variant_sandbox.spec.complexity import _RULE_COSTS, measure_complexity

        assert _RULE_COSTS["npcs"] == _RULE_COSTS["territory"] == 2.5
        spec = _boar(chess_spec)
        without = copy.deepcopy(spec)
        del without["rules"]["npcs"]
        charged = measure_complexity(spec).special_rules
        assert charged - measure_complexity(without).special_rules == 2.5

    def test_a_hunter_a_wall_and_a_free_one_are_three_different_games(
        self, chess_spec: dict[str, Any]
    ) -> None:
        from chess_variant_sandbox.spec.distance import distance, features

        boar = _boar(chess_spec)
        tame = copy.deepcopy(boar)
        tame["pieces"][-1]["captures"] = []
        free = copy.deepcopy(boar)
        free["rules"]["npcs"][0]["bounty"] = 0
        del free["rules"]["scoring"]
        assert features(boar)["has_npcs"] == features(boar)["npcs_bite"] == 1.0
        assert features(boar)["npcs_pay"] == 1.0
        assert features(tame)["npcs_bite"] == 0.0
        assert features(free)["npcs_pay"] == 0.0
        assert distance(boar, tame) > 0.0 and distance(boar, free) > 0.0
        type_only = copy.deepcopy(boar)
        del type_only["rules"]["npcs"]
        del type_only["rules"]["scoring"]
        assert distance(boar, type_only) > 0.0

    def test_the_operator_looses_a_patrol_and_takes_it_back(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """Both halves, valid both ways, and the second half undoes the first."""
        from chess_variant_sandbox.loop.operators import OPERATORS, _set_a_patrol

        assert "set-a-patrol" in {operator.name for operator in OPERATORS}
        spec = copy.deepcopy(chess_spec)
        before = spec_hash(spec)
        summary = _set_a_patrol(spec, random.Random(3))
        assert summary and "wild" in summary
        assert validate_spec(spec) == [], validate_spec(spec)
        npcs = spec["rules"]["npcs"]
        assert len(npcs) == 1 and npcs[0]["mirror"] is True and len(npcs[0]["route"]) == 4
        assert spec["schema_version"] == 9
        assert _set_a_patrol(spec, random.Random(3))
        assert validate_spec(spec) == []
        assert spec_hash(spec) == before, "set-a-patrol does not undo itself"

    def test_the_operator_refuses_where_the_validator_would(
        self, chess_spec: dict[str, Any]
    ) -> None:
        from chess_variant_sandbox.loop.operators import _set_a_patrol

        spec = copy.deepcopy(chess_spec)
        spec["rules"]["turn"] = {"moves": 2, "ramp": [1]}
        assert _set_a_patrol(spec, random.Random(3)) is None

    def test_scoring_the_clock_counts_points_when_there_is_something_wild(
        self, chess_spec: dict[str, Any]
    ) -> None:
        from chess_variant_sandbox.spec.transform import clear_npcs, set_npc, set_scoring

        spec = copy.deepcopy(chess_spec)
        set_npc(spec, "N", ["c3", "c4", "d4", "d3"])
        set_scoring(spec)
        assert spec["rules"]["scoring"] == {"count": "points"}
        assert validate_spec(spec) == []
        clear_npcs(spec)
        assert "scoring" not in spec["rules"], "points with nothing to count"

    def test_the_proposer_is_told_the_point_of_it(self) -> None:
        from chess_variant_sandbox.llm.propose import SCHEMA_PREAMBLE

        assert "`rules.npcs`" in SCHEMA_PREAMBLE
        assert '"points"' in SCHEMA_PREAMBLE

    def test_fairy_stockfish_refuses_it_by_name(self, chess_spec: dict[str, Any]) -> None:
        from chess_variant_sandbox.fairy.variant import Untranslatable, translate

        with pytest.raises(Untranslatable) as caught:
            translate(_boar(chess_spec))
        assert any("rules.npcs" in reason for reason in caught.value.reasons)


# ------------------------------------------------------------------------- the two oracles


class TestTheFuzzOracles:
    """Run against a compiled spec and a hand-built state, since no engine can play one."""

    def _engine(self, chess_spec: dict[str, Any]) -> Any:
        compiled = compile_spec(_boar(chess_spec))
        return SimpleNamespace(spec=compiled)

    def test_the_opening_position_obeys_both(self, chess_spec: dict[str, Any]) -> None:
        engine = self._engine(chess_spec)
        state = SimpleNamespace(board=engine.spec.initial_board, points=(0.0, 0.0))
        fuzz.every_npc_stands_on_its_own_route(engine, state)
        fuzz.the_points_are_the_bounties_of_the_npcs_that_are_gone(engine, state)

    def test_a_wild_piece_off_its_route_is_caught(self, chess_spec: dict[str, Any]) -> None:
        engine = self._engine(chess_spec)
        board = bytearray(engine.spec.initial_board)
        board[_square("e4")] = board[_square("c3")]
        board[_square("c3")] = 0
        with pytest.raises(fuzz.OracleFailure, match="no route runs"):
            fuzz.every_npc_stands_on_its_own_route(engine, SimpleNamespace(board=bytes(board)))

    def test_two_wild_pieces_on_one_loop_are_caught(self, chess_spec: dict[str, Any]) -> None:
        engine = self._engine(chess_spec)
        board = bytearray(engine.spec.initial_board)
        board[_square("d4")] = board[_square("c3")]
        with pytest.raises(fuzz.OracleFailure, match="one route"):
            fuzz.every_npc_stands_on_its_own_route(engine, SimpleNamespace(board=bytes(board)))

    def test_the_tally_has_to_account_for_every_missing_wild_piece(
        self, chess_spec: dict[str, Any]
    ) -> None:
        engine = self._engine(chess_spec)
        board = bytearray(engine.spec.initial_board)
        board[_square("c6")] = 0  # the second boar has been taken
        gone = bytes(board)
        fuzz.the_points_are_the_bounties_of_the_npcs_that_are_gone(
            engine, SimpleNamespace(board=gone, points=(3.0, 0.0))
        )
        fuzz.the_points_are_the_bounties_of_the_npcs_that_are_gone(
            engine, SimpleNamespace(board=gone, points=(0.0, 3.0))
        )
        with pytest.raises(fuzz.OracleFailure, match="missing wild pieces are worth 3"):
            fuzz.the_points_are_the_bounties_of_the_npcs_that_are_gone(
                engine, SimpleNamespace(board=gone, points=(0.0, 0.0))
            )

    def test_a_state_with_no_tally_fails_loudly(self, chess_spec: dict[str, Any]) -> None:
        """The intermediate state Q9b must not leave: the refusal lifted, the tally absent."""
        engine = self._engine(chess_spec)
        with pytest.raises(fuzz.OracleFailure, match="no tally"):
            fuzz.the_points_are_the_bounties_of_the_npcs_that_are_gone(
                engine, SimpleNamespace(board=engine.spec.initial_board)
            )

    def test_both_are_free_where_nothing_is_wild(self, chess_spec: dict[str, Any]) -> None:
        engine = SimpleNamespace(spec=compile_spec(chess_spec))
        state = SimpleNamespace(board=engine.spec.initial_board)
        fuzz.every_npc_stands_on_its_own_route(engine, state)
        fuzz.the_points_are_the_bounties_of_the_npcs_that_are_gone(engine, state)


# ------------------------------------------------------- the third owner is not the opponent


def _without_the_wild(engine: Engine, state: Any) -> Any:
    """The same position with every wild piece lifted off the board.

    Not a legal position under the rule -- a boar is always somewhere on its route -- and
    it is not asked to be one. It is the control the board-reading terms are measured
    against: whatever a wild piece is to them, it has to be the same thing it would be if
    it were not there, or it is being counted as somebody's.
    """
    board = bytearray(state.board)
    for square, value in enumerate(board):
        if value and engine.spec.owner_of(value) == WILD:
            board[square] = 0
    return replace(state, board=bytes(board))


class TestAWildPieceIsNobodys:
    """Q41: two files decoded an owner into a table with two rows.

    ``(value - 1) // side_stride`` is 2 for a wild piece and the tables it indexed had a
    row per player, so half these sites raised and the other half -- ``advance``,
    ``centrality``, and the material sums under them -- read a boar as **the opponent**,
    which is the half that would have measured a schema-9 showcase wrong and reported
    nothing. What is asserted here is one property per file and one decision per term: a
    wild piece is in neither army, so it enters neither side of a difference and neither
    denominator, and what it is worth to a player is its bounty.
    """

    def test_every_feature_term_is_antisymmetric(self, chess_spec: dict[str, Any]) -> None:
        """The property the silent half broke, and the reason it was silent.

        Each term is a difference between the two players, so reading it from the other
        seat must negate it exactly. A third owner tested with ``owner == player`` is
        "not me" to *both* seats, so a boar was subtracted from each of them and the
        vector stopped being antisymmetric -- no exception, no wrong-looking number, and
        a fit built on it would have been wrong about every position.
        """
        engine = Engine(_boar(chess_spec))
        evaluator = FeatureEvaluator.for_engine(engine, (1.0,) * len(VOCABULARY))
        state = replace(engine.initial_state(), points=(3.0, 0.0))
        mine = evaluator.features(engine, state, 0)
        theirs = evaluator.features(engine, state, 1)
        assert mine == tuple(-value for value in theirs)

    def test_the_board_reading_terms_read_the_two_armies_and_nothing_else(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """Lifting the boars off the board moves no term that reads the board.

        ``material``, ``advance`` and ``centrality`` are statements about the two armies;
        a piece in neither is in neither the numerator nor the average. The terms that
        *do* price a boar are ``mobility`` and ``royal_exposure``, and both of them ask
        the engine rather than decoding a byte here -- which is where an obstacle that
        blocks and threatens belongs.
        """
        engine = Engine(_boar(chess_spec))
        evaluator = FeatureEvaluator.for_engine(engine, (1.0,) * len(VOCABULARY))
        state = engine.initial_state()
        with_boars = evaluator.features(engine, state, 0)
        without = evaluator.features(engine, _without_the_wild(engine, state), 0)
        for term in ("material", "advance", "centrality"):
            index = VOCABULARY.index(term)
            assert with_boars[index] == without[index], term

    def test_a_boar_is_nobodys_material(self, chess_spec: dict[str, Any]) -> None:
        """``agents/evaluation`` says the same thing about the same board."""
        engine = Engine(_boar(chess_spec))
        objective = Evaluator.for_spec(engine.spec)
        state = engine.initial_state()
        bare = _without_the_wild(engine, state)
        assert objective.material(state, 0) == objective.material(bare, 0)
        assert objective.material(state, 0) == -objective.material(state, 1)

    def test_a_boar_is_worth_its_bounty_and_the_bounty_is_a_tally(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """``scoring.count: "points"`` counts what is *gone*, so it cannot read the board.

        Falling through to the piece count -- which is what both evaluators did -- made a
        player's score sixteen at ply zero and never moved it, in the one rule schema 9
        added for the sake of wild pieces.
        """
        engine = Engine(_boar(chess_spec))
        objective = Evaluator.for_spec(engine.spec)
        opening = engine.initial_state()
        assert objective.score_for(opening, 0) == 0.0
        paid = replace(opening, points=(3.0, 0.0))
        assert objective.score_for(paid, 0) == 3.0
        assert objective.score_for(paid, 1) == 0.0

        context = feature_context(engine.spec)
        assert _score_gap(paid, context, 0) == 3.0
        # Every bounty standing on the board at ply zero: two boars at three apiece.
        assert context.score_ceiling == 6.0

    def test_the_opening_material_counts_the_two_armies(
        self, chess_spec: dict[str, Any]
    ) -> None:
        """The denominator has to be the quantity the numerator is drawn from.

        ``total_material`` normalises ``material``, and ``material`` no longer counts a
        boar; counting one here would price both armies against a total neither of them
        can ever hold.
        """
        plain = feature_context(compile_spec(chess_spec))
        wild = feature_context(compile_spec(_boar(chess_spec)))
        assert wild.total_material == plain.total_material
        assert wild.initial_counts[: plain.n_types] == plain.initial_counts
        assert wild.initial_counts[wild.n_types - 1] == 0, "the boar type opens with none"


# ------------------------------------------------ Q922: what a wild piece is worth by default


class TestAWildPieceWithNoBountyPaysTheValueItDeclares:
    @staticmethod
    def _pays(piece: dict[str, Any], npc: dict[str, Any]) -> float:
        from chess_variant_sandbox.spec.distance import _npc_features

        return _npc_features([{"id": "X"}, piece], [{"piece": piece["id"], **npc}])["npcs_pay"]

    def test_no_bounty_and_no_value_is_the_value_every_piece_has(self) -> None:
        assert self._pays({"id": "W"}, {}) == 1.0

    @pytest.mark.parametrize(
        ("piece", "npc", "pays"),
        [
            ({"id": "W", "value": 0}, {}, 0.0),
            ({"id": "W", "value": 4}, {}, 1.0),
            ({"id": "W", "value": 4}, {"bounty": 0}, 0.0),
            ({"id": "W", "value": 0}, {"bounty": 2}, 1.0),
            ({"id": "W"}, {"bounty": True}, 0.0),
            ({"id": "W"}, {"bounty": "lots"}, 0.0),
        ],
        ids=str,
    )
    def test_the_bounty_beats_the_value(self, piece: dict, npc: dict, pays: float) -> None:
        assert self._pays(piece, npc) == pays


# ------------------------------------------------ Q922: the named-square and wild-piece checks


class TestANameThatIsNotASquareDoesNotEndTheList:
    """Each reader skips a malformed name and goes on to the next, which is how the
    validator can report every problem in a draft at once."""

    BAD = ["z9", 3, None, "a0", ""]

    def test_walls(self) -> None:
        from chess_variant_sandbox.spec.schema import obstacle_squares

        rules = {"obstacles": {"squares": [*self.BAD, "b2"]}}
        assert obstacle_squares(rules, 3, 3) == {1 * 3 + 1}

    def test_a_wild_pieces_route(self) -> None:
        from chess_variant_sandbox.spec.schema import npc_routes

        rules = {"npcs": [{"piece": "W", "route": [*self.BAD, "a1", "b1"]}]}
        assert npc_routes(rules, 3, 3) == [("rules.npcs[0]", (0, 1))]

    def test_where_a_piece_starts_and_where_it_is_mirrored_to(self) -> None:
        from chess_variant_sandbox.spec.schema import _placed_ranks

        setup = {"placements": [{"piece": "K", "squares": [*self.BAD, "a1"]}]}
        assert _placed_ranks(setup, "K", 3, 4, "mirror") == {0, 3}
        assert _placed_ranks({**setup, "mirror": False}, "K", 3, 4, "mirror") == {0}
        assert _placed_ranks({**setup, "mirror": True}, "K", 3, 4, "rotate") == {0, 3}
        assert _placed_ranks(setup, "Q", 3, 4, "mirror") == set()
        assert _placed_ranks({"placements": "a1"}, "K", 3, 4, "mirror") == set()


class TestNamedSquaresAgainstWalls:
    """`_check_named_squares` on a 3x3 board whose wall (a3) is the mirror image of a1."""

    @staticmethod
    def _errors(setup: dict[str, Any], absent: bool = False) -> list[str]:
        from chess_variant_sandbox.spec.schema import _check_named_squares

        errors: list[str] = []
        _check_named_squares({}, setup, 3, 3, "mirror", {6}, absent, errors)
        return errors

    @pytest.mark.parametrize(
        ("setup", "refused"),
        [
            ({"placements": [{"piece": "P", "squares": ["a3"]}]}, True),            # on the wall
            ({"placements": [{"piece": "P", "squares": ["a1"]}]}, True),            # its image, by default
            ({"placements": [{"piece": "P", "squares": ["a1"]}], "mirror": True}, True),
            ({"placements": [{"piece": "P", "squares": ["a1"]}], "mirror": False}, False),
            ({"placements": [{"piece": "P", "squares": ["b1"]}]}, False),
            ({"placements": [{"piece": "P", "squares": ["z9", "a3"]}]}, True),      # junk, then the wall
            ({"placements": ["junk", {"piece": "P", "squares": ["a3"]}]}, True),
            ({"placements": "a3"}, False),
        ],
        ids=str,
    )
    def test_a_placement(self, setup: dict[str, Any], refused: bool) -> None:
        assert bool(self._errors(setup)) is refused

    def test_a_hexagons_missing_corner_is_said_in_other_words(self) -> None:
        setup = {"placements": [{"piece": "P", "squares": ["a3"]}]}
        assert "not cells of this board" in self._errors(setup, absent=True)[0]
        assert "nothing may occupy a wall" in self._errors(setup)[0]


class TestWildPiecesAgainstTheRestOfTheSpec:
    @staticmethod
    def _errors(rules: dict[str, Any], pieces: list[Any] | None = None, **spec: Any) -> list[str]:
        from chess_variant_sandbox.spec.schema import _check_npc_coherence

        whole = {"board": {"width": 3, "height": 3}, **spec}
        errors: list[str] = []
        _check_npc_coherence(whole, pieces if pieces is not None else [], rules, errors)
        return errors

    POINTS = {"scoring": {"count": "points"}}

    @pytest.mark.parametrize("npcs", [None, [], {}, "wolves", {"piece": "W"}], ids=str)
    def test_points_with_no_wild_piece_list_count_nothing(self, npcs: Any) -> None:
        rules = {**self.POINTS, **({} if npcs is None else {"npcs": npcs})}
        (error,) = self._errors(rules)
        assert error.startswith('rules.scoring.count "points" needs rules.npcs')

    def test_junk_among_the_pieces_is_not_a_piece(self) -> None:
        rules = {"npcs": [{"piece": "W", "route": ["a1", "b1"]}]}
        assert self._errors(rules, [{"x": 1}, "grid", None, {"id": "W"}]) == []

    def test_a_shared_square_is_named_by_its_file_and_rank(self) -> None:
        rules = {"npcs": [{"piece": "W", "route": ["a1", "c2"]}, {"piece": "W", "route": ["c2", "b3"]}]}
        (error,) = self._errors(rules)
        assert error.startswith("rules.npcs[1] and rules.npcs[0] both walk c2;")

    @pytest.mark.parametrize(
        ("setup", "refused"),
        [
            ({"placements": [{"piece": "P", "squares": ["a1"]}]}, True),                  # under it
            ({"placements": [{"piece": "P", "squares": ["a3"]}]}, True),                  # its image
            ({"placements": [{"piece": "P", "squares": ["a3"]}], "mirror": False}, False),
            ({"placements": [{"piece": "P", "squares": ["z9", 1, "a1"]}]}, True),
            ({"placements": ["junk", {"piece": "P"}, {"piece": "P", "squares": ["a1"]}]}, True),
            ({"placements": [{"piece": "P", "squares": ["c3"]}]}, False),
        ],
        ids=str,
    )
    def test_a_start_under_the_array(self, setup: dict[str, Any], refused: bool) -> None:
        rules = {"npcs": [{"piece": "W", "route": ["a1", "b1"]}]}
        errors = self._errors(rules, setup=setup)
        assert bool(errors) is refused
        assert all("needs an empty square to start from" in e for e in errors)

    @pytest.mark.parametrize(
        ("npc", "piece", "pays"),
        [
            ({}, {"id": "W"}, True),                         # no bounty, no value: one point
            ({}, {"id": "W", "value": 3}, True),
            ({}, {"id": "W", "value": 0}, False),
            ({}, {"id": "W", "value": 0.5}, True),
            ({}, {"id": "W", "value": "heavy"}, True),       # not a number: the default
            ({}, {"id": "V"}, True),                         # the piece is not declared: the default
            ({"bounty": 0}, {"id": "W", "value": 5}, False),
            ({"bounty": 2}, {"id": "W", "value": 0}, True),
            ({"bounty": True}, {"id": "W", "value": 0}, False),
        ],
        ids=str,
    )
    def test_what_a_wild_piece_pays(self, npc: dict, piece: dict, pays: bool) -> None:
        rules = {**self.POINTS, "npcs": [{"piece": "W", "route": ["a1", "b1"], **npc}]}
        errors = self._errors(rules, [piece])
        assert bool(errors) is not pays
        assert all("needs a wild piece with a bounty above zero" in e for e in errors)
