# Piece icons

A library of freely-licensed SVG piece icons, for anything that wants to draw a
position rather than read one.

```
assets/pieces/
  svg/               the icons themselves
  manifest.json      slug -> file, viewBox, source URL, author, licence
  aliases.json       ruleset piece names -> slugs, hand-maintained
  catalogue.json     slug -> name, category, search words; the designer's palette, generated
  CREDITS.md         per-file attribution, generated
  index.html         contact sheet, generated
  draw.py            draws the library's own pieces in Cburnett's grammar into svg/
  fetch.py           rebuilds svg/, manifest.json and CREDITS.md from Commons
  catalogue.py       rebuilds catalogue.json from manifest.json and draw.py
  contact_sheet.py   rebuilds index.html from catalogue.json and manifest.json
  check.py           sorts the library by silhouette, checks black on the dark
                     square, and (with `frames`) flags any icon that draws wrong
```

Everything in `svg/` is committed, so nothing here runs at build time. `fetch.py`
is provenance: it records which Wikimedia Commons file each icon came from and
re-checks each licence against an allowlist.

214 files, 114 drawings: 12 standard chess files, 154 fairy (128 of them drawn here:
122 by `draw.py` in Cburnett's grammar, and the six rock, paper and scissors before it), 6
inverted, 14 shogi, 28 xiangqi. The designer's palette offers them by category:
leapers, riders, compounds, royals, pawns, oddities, and the two families kept apart,
shogi and xiangqi (`catalogue.json`).

## Naming

Two-colour pieces are `<slug>-white.svg` and `<slug>-black.svg`. The shogi tiles
are the exception: that set distinguishes the two players by rotating a piece
rather than recolouring it, so `shogi-gold.svg` is one file and the second player
is a 180° rotation. Xiangqi *does* recolour, and also uses a different character
per side (帥 for red, 將 for black), so it has real pairs — `-white` is the red
side. It comes in two forms: `xiangqi-cannon-*` is the Han character and
`xiangqi-pictorial-cannon-*` the western-readable drawing on the same disc.

The western pieces are Cburnett's set and its fairy extensions: one style, 45×45,
transparent background.

## The grammar (plan-visual.md §7, decision D4)

`draw.py` draws 61 pieces from a few shared parts so that a board of fairy pieces reads
as one set with the familiar six: a 45 × 45 frame, a 1.5 black stroke with round joins,
the first player's piece white and the second's black, a detail line black on a white
piece and white on a black one, one shared base, transparent ground. A piece is a *body*
on that base and a *head* on the body. The body says what kind of piece it is (a leaper's
short stem and collar, a rider's straight shaft, a compound's broad column under the
rook's battlements, the bishop's mitre or the queen's coronet, the royal barrel, the
pawn's skirt, or for an oddity the base alone) and the head says which one. A grown head
keeps the one stroke width (`Mark.k`). Where two parts meet a seam is drawn, invisible on
a white piece and the white line that keeps a black piece from reading as one blot.

Every part is original geometry, released CC0, except Cburnett's knight head, which the
knight's family (zebra, nightrider, unicorn, dragon, royal knight) wears on the grammar's
bodies; those keep his CC BY-SA 3.0 and credit him. They replace Commons' zebra,
nightrider, unicorn and dragon, each a whole knight with a mark on it that the sort below
could not tell from the knight. Edit `draw.py`, never a drawn file: a test holds every file
to what the script writes.

**The sort.** No two western pieces may look alike at 24 px. `check.py` rasterises each
piece's white form there with resvg, takes its outline, and scores two outlines by the
boundary F-score at one pixel; the threshold is the familiar set's own worst pair, bishop
and pawn (0.947), which every chess player tells apart. Shogi tiles and xiangqi discs are
one outline by design and are not sorted. Each drawing's head scale in `draw.py` was
chosen so the whole library sorts; a pair over the threshold is a drawing to change, not
a threshold to move (plan-visual.md §7 has the rule as it was fixed before any measuring,
and its amendment). The same script holds every black piece's body at 4.5 : 1 or more on
the board's dark square under its worst streak, in both themes; the least today is the
ship, 6.2 : 1. The chancellor is drawn here from two of his pieces, the
rook's tower with the knight's head on its battlements, because Commons' own
chancellor (a whole knight on a sliver of upturned battlements) read as a knight
to a first-time visitor; the archbishop, by the same rule, is his bishop's mitre
with the knight's head rising from behind its point, where Commons' is a whole
knight on a bishop's foot. The alfil (an elephant's head), the camel (standing, one
hump) and the wazir (a ring crossed by a plus, the fairy fonts' disc) are drawn in the
grammar too; until Q557b they were Quivira's font glyphs, a second style. The shogi set
is Hari Seldon's westernised tiles; the xiangqi set is Inductiveload's discs. Mixing the families on one board will look
like several sets, because it is.

Slugs describe the *drawing*, not the Commons filename. Commons calls
`Chess_w{l,d}t45.svg` a wizard; it is unambiguously a crescent, so here it is
`moon`. Where the two disagree, the picture wins.

## Using them

`manifest.json` carries a `viewBox` for every file, because the sources are not
all 45×45. Scale to that box rather than to a pixel size:

```html
<svg viewBox="0 0 45 45" width="48" height="48"><use href="svg/knight-white.svg#..."/></svg>
```

Inlining is usually easier than `<use>` — see `contact_sheet.py:inline`, which
strips the XML prologue and forces a known `viewBox` so a source file's hardcoded
`width` cannot override the CSS.

`aliases.json` maps the piece names that appear in `rulesets/` onto slugs, so a
renderer can pick an icon without hardcoding chess. It is a judgement about what
a piece *looks* like, not what it does: `advisor` and `met` both draw as a ferz
because that is the shape, not because the DSL says they move alike. An unmapped
name is not an error — fall back to the spec letter.

## Who reads it

`chess_variant_sandbox/viz/pieces.py`, and only for drawing. It looks a piece name up in
`aliases.json`, falls back to the name itself as a slug, and returns `None` if
neither names a file — so the replay page draws an icon where the library has one
and the spec letter where it does not, which is the ordinary case for a piece
proposed twenty minutes ago. `chess-variant-sandbox viz --letters` turns icons off entirely,
because a borrowed chess glyph quietly implies that the piece moves like the chess
piece it resembles, and that is worth suppressing when the point of the replay is a
proposal rather than a known game.

Icons reach the page as `data:` URIs rather than as shared `<symbol>` elements: four
files carry their own `<style>` block with class names like `.s0` and several reuse
ids like `path1`, so one document per icon is what keeps them from restyling each
other. It also means the bytes travel unmodified, which is what the licences ask
for. The page carries the attribution its icons require in a footer, from
`manifest.json` where that exists and as a pointer to `CREDITS.md` where it does not.

The designer's palette (`web/piece_library.py`, served at `GET /api/design?part=pieces`
and fetched when the palette first opens) reads `catalogue.json` and the first seat's file
of each drawing, so a designer can search the library by name, kind or move and dress a
piece in a drawing; the slug picked goes into `appearance.icons`, outside the mechanics
hash. Nothing else in `chess_variant_sandbox/` reads this directory — not the engine, not
the tournament, not the metrics.

## Licensing

Every file is under a free licence, and `fetch.py` fails rather than downloading
anything outside its allowlist. As it stands: 124 CC0, 56 CC BY-SA 3.0, 28 public
domain, 5 CC BY-SA 4.0, 1 GFDL. `CREDITS.md` names the author, licence and source
page for each one.

The 122 files drawn in the grammar are described above: 112 CC0, and the 10 of the
knight's family that wear Cburnett's head under his CC BY-SA 3.0. Of the other
non-xiangqi files, 54 are redistributed
byte-for-byte unmodified; the
two chancellor files are Cburnett's rook and knight rearranged and the two
archbishop files his bishop and knight, under his CC BY-SA 3.0, and the six rock, paper and scissors files are originals released CC0. The 28
xiangqi files are each one disc cut out of a four-up source sheet; that source is
public domain, so the cut carries no obligation, and `CREDITS.md` records it in
its notes column anyway.

The CC BY-SA files carry a share-alike obligation: if you modify one and
distribute it, the modified file has to stay under the same licence. Attribution
is required for them too, which is what `CREDITS.md` is for — ship it alongside
anything that ships the icons.

## Rebuilding

```bash
uv run python assets/pieces/draw.py            # the grammar's pieces
python assets/pieces/fetch.py                  # or `--local`, without the network
uv run python assets/pieces/catalogue.py
uv run python assets/pieces/contact_sheet.py
uv run python assets/pieces/check.py           # the sort and the contrast
uv run python assets/pieces/check.py frames    # and the frames, in headless Chrome
```

`check.py frames` renders every icon in headless Chrome and compares the bounding box
of the art against the frame it is supposed to fill, flagging anything clipped,
empty, or lost in a corner. It is separate from `fetch.py` because it needs a
browser and the fetch must work without one. It earns its place: a bug in
`contact_sheet.inline` left every Inkscape-authored file un-normalised, because
the pattern matching the opening `<svg>` tag was anchored to the start of the
file and Inkscape writes a comment ahead of it. The page still rendered, the
files still parsed, and nothing else noticed.

What it does not catch is art that draws correctly and simply looks wrong — the
Commons nightrider `Chess_N{l,d}t45.svg` fills its frame properly and is still a
malformed horse, which is why the library uses `Chess_K{l,d}t45.svg` instead.
That one needs an eye on `index.html`.

`fetch.py` skips files already on disk, so an interrupted run resumes. Delete
`svg/` for a clean rebuild. It is deliberately slow: `upload.wikimedia.org`
tolerates a short burst and then rate-limits the IP for several minutes, so
downloads are spaced by `THROTTLE` seconds and a 429 is waited out rather than
retried tightly.
