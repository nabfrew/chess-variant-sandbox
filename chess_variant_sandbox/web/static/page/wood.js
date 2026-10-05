// The board's wood (plan-visual.md §9, the third answer; Q588).
//
// Every square is its own piece of wood: a window cut from one of two grain tiles, the light
// squares' or the dark squares' (`--grain-lt`, `--grain-dk`, set in `web/page/tokens.py`),
// at an offset of its own. The offset is a hash of the board's identity and the square's
// index, never a live random draw, so a board looks the same on every load, in every
// screenshot and after every redraw, and a square keeps its piece of wood when the board
// is flipped. The frame round the board is walnut (`--walnut`, `style.py`), started at an
// offset hashed from the same identity.
//
// It draws nothing a rule means: the grain is laid over the square's own colour and under
// every highlight, mark and piece, at the theme's `--grain-lt-a` / `--grain-dk-a` opacity,
// so a theme switch changes how strongly it shows without a redraw. Both boards use it, the
// play board (`board.js`) and the still board (`stillboard.js`), so there is one way a
// square is grained.

// A tile's side, in cells: how much of the grain one board's worth of squares draws from.
// The dark squares' tile is larger as well as coarser, so the two woods differ in scale on
// screen and not only in the tile.
const WOOD_TILE = { lt: 3.2, dk: 4.4 };

// FNV-1a over the identity's text: a game's hash on the play board, the still board's own
// description of itself.
function woodSeed(identity) {
  const text = String(identity == null ? '' : identity);
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

// A well-mixed 32-bit number from a seed and a counter (the finaliser of MurmurHash3).
function woodHash(seed, n) {
  let h = (seed ^ Math.imul(n + 1, 0x9e3779b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

// Where in its tile a square's piece of wood is cut from, as two fractions in [0, 1).
function woodOffset(seed, square) {
  return [woodHash(seed, 2 * square) / 4294967296, woodHash(seed, 2 * square + 1) / 4294967296];
}

// The two tiles, read once from the stylesheet. `null` where there is no stylesheet to read
// (a test's stand-in document), and then the board is drawn plain.
let WOOD_TILES;
function woodTiles() {
  if (WOOD_TILES !== undefined) return WOOD_TILES;
  WOOD_TILES = null;
  try {
    const css = getComputedStyle(document.documentElement);
    const read = name => {
      // `tokens._uri` quotes the tile in double quotes and has none inside it.
      const found = /url\("(data:[^"]*)"\)/.exec(css.getPropertyValue(name) || '');
      return found ? found[1] : null;
    };
    const lt = read('--grain-lt'), dk = read('--grain-dk');
    if (lt && dk) WOOD_TILES = { lt: lt, dk: dk };
  } catch (e) { WOOD_TILES = null; }
  return WOOD_TILES;
}

// Which wood a cell is cut from, read off the colour `cellShade` gave it: the light
// squares' maple for the light shade, the dark squares' for the dark one and for a
// hexagon's third, in-between shade.
function woodTone(shade) {
  return shade === 'var(--light)' ? 'lt' : 'dk';
}

// One board's grain. `tag` keeps two boards on one page apart; `identity` is what the
// offsets are hashed from; `cell` is the board's cell size in its own units. `cell(...)`
// returns the grain for one drawn cell (a ghost copy is drawn at its own place, so it gets
// its own pattern, at the same offset as its square), and `defs()` the patterns they read,
// to be put anywhere in the same `<svg>`.
function woodGrain(tag, identity, cell) {
  const tiles = woodTiles();
  const seed = woodSeed(identity);
  const id = tag + '-wood';
  let defs = '';
  let count = 0;
  const used = { lt: false, dk: false };
  const n = v => Number(v.toFixed(3));
  return {
    cell(x, y, view, square, shade) {
      if (!tiles) return '';
      const tone = woodTone(shade);
      const side = WOOD_TILE[tone] * cell;
      const at = woodOffset(seed, square);
      const name = `${id}${++count}`;
      used[tone] = true;
      defs += `<pattern id="${name}" href="#${id}-${tone}" patternTransform="translate(${
        n(x - at[0] * side)} ${n(y - at[1] * side)})"/>`;
      return cellShape(x, y, cell, view,
        `fill="url(#${name})" class="grain ${tone}" pointer-events="none"`);
    },
    defs() {
      if (!tiles) return '';
      let base = '';
      for (const tone of ['lt', 'dk']) {
        if (!used[tone]) continue;
        const side = n(WOOD_TILE[tone] * cell);
        base += `<pattern id="${id}-${tone}" patternUnits="userSpaceOnUse" width="${side}"
          height="${side}"><image href="${tiles[tone]}" width="${side}" height="${side}"
          preserveAspectRatio="none"/></pattern>`;
      }
      return `<defs>${base}${defs}</defs>`;
    },
  };
}

// Where the frame's walnut starts, in px into its 240 px tile: hashed from the same
// identity, so a board's frame is its own as well as its squares. `woodFrameStyle` is the
// same as a `style` value, for a frame drawn as a string.
function woodFrameAt(identity) {
  const seed = woodSeed('frame:' + String(identity == null ? '' : identity));
  return [woodHash(seed, 0) % 240, woodHash(seed, 1) % 240];
}

function woodFrameStyle(identity) {
  const at = woodFrameAt(identity);
  return `--wood-x: ${at[0]}px; --wood-y: ${at[1]}px`;
}

// A library card's own sheet of paper (Q684): the same fibre tile every panel is laid with
// (`--paper`), cut at an offset of its own, a touch larger or smaller, turned a little and
// a touch lighter or darker, so a grid of cards reads as separate sheets laid on a table.
// Hashed from the game's hash like a square's wood, never drawn at random, so a card is the
// same sheet on every load and every machine. The ranges are `tokens.SHEET`'s, and kept
// small on purpose: offset anywhere in the 180 px tile, size within 6%, angle within 3
// degrees, tone within one unit either way (`style.py` turns it into .6% of lightness and
// .2% of chroma, as far as the contrast floors allow). It returns a `style` value.
const SHEET = { tile: 180, scale: 0.06, angle: 3 };

function paperSheet(identity) {
  const seed = woodSeed('sheet:' + String(identity == null ? '' : identity));
  const unit = n => woodHash(seed, n) / 4294967296;
  const signed = n => 2 * unit(n) - 1;
  const scale = 1 + SHEET.scale * signed(2);
  const n = (v, places) => Number(v.toFixed(places));
  return `--sheet-x: ${n(unit(0) * SHEET.tile * scale, 1)}px; --sheet-y: ${
    n(unit(1) * SHEET.tile * scale, 1)}px; --sheet-s: ${n(scale, 3)}; --sheet-a: ${
    n(SHEET.angle * signed(3), 2)}deg; --sheet-t: ${n(signed(4), 2)}`;
}
