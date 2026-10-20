# Piece icons

A library of freely-licensed SVG piece icons, for anything that wants to draw a
position rather than read one.

```
assets/pieces/
  svg/               the icons themselves
  manifest.json      slug -> file, viewBox, source URL, author, licence
  aliases.json       ruleset piece names -> slugs, hand-maintained
  CREDITS.md         per-file attribution, generated
  index.html         contact sheet, generated
  fetch.py           rebuilds svg/, manifest.json and CREDITS.md from Commons
  contact_sheet.py   rebuilds index.html from manifest.json
  check.py           renders every icon and flags any that draws wrong
```

Everything in `svg/` is committed, so nothing here runs at build time. `fetch.py`
is provenance: it records which Wikimedia Commons file each icon came from and
re-checks each licence against an allowlist.

106 files: 12 standard chess, 46 fairy (six of them rock, paper and scissors), 6
inverted, 14 shogi, 28 xiangqi.

## Naming

Two-colour pieces are `<slug>-white.svg` and `<slug>-black.svg`. The shogi tiles
are the exception: that set distinguishes the two players by rotating a piece
rather than recolouring it, so `shogi-gold.svg` is one file and the second player
is a 180° rotation. Xiangqi *does* recolour, and also uses a different character
per side (帥 for red, 將 for black), so it has real pairs — `-white` is the red
side. It comes in two forms: `xiangqi-cannon-*` is the Han character and
`xiangqi-pictorial-cannon-*` the western-readable drawing on the same disc.

The western pieces are Cburnett's set and its fairy extensions: one style, 45×45,
transparent background. The chancellor is drawn here from two of his pieces, the
rook's tower with the knight's head on its battlements, because Commons' own
chancellor (a whole knight on a sliver of upturned battlements) read as a knight
to a first-time visitor; the archbishop, by the same rule, is his bishop's mitre
with the knight's head rising from behind its point, where Commons' is a whole
knight on a bishop's foot. Alfil, camel and wazir are Quivira glyphs, a second and
slightly different style. A font's white piece is an outline with nothing inside it,
so the three white ones here have that body filled white beneath the outline
(`fetch.py`'s `FILLED`); as downloaded, a white elephant on a board was the square
showing through a black line. The shogi set is Hari Seldon's westernised tiles; the
xiangqi set is Inductiveload's discs. Mixing the families on one board will look
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

Nothing else in `chess_variant_sandbox/` reads this directory — not the engine, not the
tournament, not the metrics.

## Licensing

Every file is under a free licence, and `fetch.py` fails rather than downloading
anything outside its allowlist. As it stands: 54 CC BY-SA 3.0, 28 public domain,
11 CC BY-SA 4.0, 12 CC0, 1 GFDL. `CREDITS.md` names the author, licence and source
page for each one.

Of the 78 non-xiangqi files, 65 are redistributed byte-for-byte unmodified; the
three white Quivira glyphs are the originals with their body filled white beneath
the outline, under the same CC BY-SA 4.0; the
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
python assets/pieces/fetch.py
python assets/pieces/contact_sheet.py
python assets/pieces/check.py
```

`check.py` renders all 100 icons in headless Chrome and compares the bounding box
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
