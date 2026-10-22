// ------------------------------------------------------- a board without edges
//
// Shared by the play board and the offline replay. Pure functions of a position and a
// view offset, so what the shift does to a square and what path a crossing move takes
// can both be asserted in a test without a browser.

// How many copies of the board are drawn beyond the outline, and how much of them the
// frame shows. The outermost ring runs off the edge of the frame rather than stopping
// at it, which is what makes the repetition read as endless instead of as a border.
// Capped against the board's own size, so a small board is not mostly ghost.
const GHOST_MARGIN = 2.4;

// Non-negative modulo. `%` keeps the sign of the dividend in JavaScript, and every
// coordinate here is a lap count that may be negative.
function wrapMod(value, size) { return ((value % size) + size) % size; }

// A style worked out while drawing -- a bar's width, a card's sheet of paper, a label's
// place -- as a class (G12). The site's content policy refuses `style=` attributes
// (`style-src 'self'`, `web/server.py`), so markup carries none; a value only the script
// knows becomes one rule in a sheet the script made, written through the CSSOM, which the
// policy does not govern: `setProperty` takes one value for one property, so nothing in a
// value can open another rule. `!important`, as the attribute stood above every selector.
// The same declarations are the same class, so a redraw adds no rule. Shared by the page
// and the offline replay, the one file both load; where there is no document to hold a
// sheet (a test's run under Node) the class is named and nothing is styled.
const STYLE_CLASSES = new Map();
let STYLE_SHEET = null;

function styleDeclarations(text) {
  const out = [];
  let depth = 0, start = 0;
  const take = end => {
    const part = text.slice(start, end);
    const colon = part.indexOf(':');
    if (colon > 0) out.push([part.slice(0, colon).trim(), part.slice(colon + 1).trim()]);
  };
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (c === '(') depth += 1;
    else if (c === ')') depth = Math.max(0, depth - 1);
    else if (c === ';' && !depth) { take(i); start = i + 1; }
  }
  take(text.length);
  return out.filter(([name, value]) => name && value);
}

function styleClass(declarations) {
  const text = String(declarations == null ? '' : declarations).trim();
  if (!text) return '';
  const known = STYLE_CLASSES.get(text);
  if (known) return known;
  const name = 'sc-' + STYLE_CLASSES.size.toString(36);
  STYLE_CLASSES.set(text, name);
  try {
    if (!STYLE_SHEET && typeof CSSStyleSheet === 'function' && typeof document !== 'undefined'
        && Array.isArray(document.adoptedStyleSheets)) {
      STYLE_SHEET = new CSSStyleSheet();
      document.adoptedStyleSheets = [...document.adoptedStyleSheets, STYLE_SHEET];
    }
    if (STYLE_SHEET) {
      const at = STYLE_SHEET.insertRule(`.${name} {}`, STYLE_SHEET.cssRules.length);
      const rule = STYLE_SHEET.cssRules[at];
      styleDeclarations(text).forEach(([property, value]) =>
        rule.style.setProperty(property, value, 'important'));
    }
  } catch (err) { /* no sheet to write into: the element is drawn unstyled */ }
  return name;
}

// Which edges a player may leave by, as `[low, high]` per axis. Player `null` asks for
// the union, which is the board's shape rather than anyone's permission -- a seam
// either side can cross is a seam, and a wild piece owns no seat.
function wrapAllow(rule, player) {
  const pick = axis => {
    const rows = (rule && rule[axis]) || [];
    if (player === 0 || player === 1) return rows[player] || [false, false];
    return [rows.some(row => row[0]), rows.some(row => row[1])];
  };
  const allow = { files: pick('files'), ranks: pick('ranks') };
  // A hexagon's copies this player may leave into, as their vectors (schema 16). Only
  // there, so a rectangle's answer is the two axes it always was.
  const copies = (rule && rule.copies) || [];
  if (copies.length) {
    allow.copies = copies.filter(copy => (player === 0 || player === 1)
      ? copy.permission[player] : copy.permission[0] || copy.permission[1])
      .map(copy => copy.vector);
  }
  return allow;
}

// Whether each axis wraps at all, which is what decides the shape of the drawing.
function wrapAxes(rule) {
  const union = wrapAllow(rule, null);
  return { files: union.files[0] || union.files[1],
           ranks: union.ranks[0] || union.ranks[1] };
}

// The frame of reference the board is being read in. `shift` is in whole squares and
// may be any integer; it is reduced here, so a caller only ever adds one to it.
//
// The outline's bottom-left corner is the viewBox origin, so every coordinate below is
// the one a board with edges already had and a non-wrapping axis is untouched.
//
// `lattice` is the payload's `"square"` or `"hex"` (schema 14). A hex view places cells
// on a sheared grid and draws them as hexagons; nothing else about the view changes.
//
// `cut` is `boardCut`'s answer for a board cut to a hexagon (schema 16), and absent on
// every other board. A hexagon's seams are not axes, so its `files` and `ranks` never
// wrap and its view never shifts: the outline stays where the frame names it.
function boardView(W, H, axes, shift, flipped, lattice, cut) {
  const resolve = (wraps, size, offset) => {
    if (!wraps) return { shift: 0, margin: 0, rings: 0 };
    const margin = Math.min(GHOST_MARGIN, size * 0.4);
    return { shift: wrapMod(offset, size), margin: margin,
             rings: Math.max(1, Math.ceil(margin)) };
  };
  const view = { W: W, H: H, flipped: !!flipped, hex: lattice === 'hex',
                 files: resolve(axes.files, W, (shift && shift.files) || 0),
                 ranks: resolve(axes.ranks, H, (shift && shift.ranks) || 0),
                 wrapped: !!(axes.files || axes.ranks) };
  if (cut) view.cut = cut;
  return view;
}

// A board with edges, which is what a still preview of a spec is drawn on. The same
// view object with nothing wrapping, so one function still places every square. A
// hexagon keeps its cut and its seams but draws no copy across them.
function flatView(W, H, flipped, lattice, cut) {
  return boardView(W, H, {}, null, flipped, lattice,
                   cut ? Object.assign({}, cut, { joined: [] }) : null);
}

// ------------------------------------------------ a board cut to a hexagon (schema 16)
//
// A hexagon is the hex rhombus with its two corners taken away. The compiler says which
// cells are gone (the payload's `absent`) and where each of the six copies of the board
// lies (`wrap_payload`'s `copies`); everything here is a lookup in those two tables, so
// the page decides nothing about the hexagon's shape but how to draw it.

// The six cells that touch, as steps in files and ranks, in the order `cellXY` places
// them round a cell: right, lower right, lower left, left, upper left, upper right.
const HEX_STEPS = [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]];

// What a view needs to draw a hexagon, or `null` for any other board. `joined` is the
// copies across a side somebody may cross, which are the ones drawn as ghosts; a copy
// across a side nobody crosses is not there, because that side is an edge.
function boardCut(absent, rule) {
  const gone = absent || [];
  const copies = (rule && rule.copies) || [];
  if (!gone.length && !copies.length) return null;
  return { absent: new Set(gone), copies: copies,
           joined: copies.filter(copy => pairWraps(rule, copy.pair)),
           marked: cutSeams(rule).barrier.length > 0 };
}

// Whether a pair of sides is a seam for anyone, either way.
function pairWraps(rule, pair) {
  return ((rule && rule[pair]) || []).some(row => row[0] || row[1]);
}

// Whether a place in the grid is a cell of the board proper.
function cellOnBoard(u, v, view) {
  if (u < 0 || v < 0 || u >= view.W || v >= view.H) return false;
  return !(view.cut && view.cut.absent.has(v * view.W + u));
}

// Whether a square of the frame is one the board does not have.
function cellAbsent(square, view) { return !!(view && view.cut && view.cut.absent.has(square)); }

// Which square is drawn at a place outside the board, and in which of `copies`: the copy
// whose vector carries a board cell there. `null` where none of them does.
function copyAt(u, v, view, copies) {
  for (const copy of copies) {
    const a = u - copy.vector[0], b = v - copy.vector[1];
    if (cellOnBoard(a, b, view)) return { copy: copy, square: b * view.W + a };
  }
  return null;
}

// The ghosts round a hexagon, found outwards from the board a ring at a time, so a
// ghost's `depth` is its distance in cells from the nearest cell of the board. The walk
// only goes through the copies across joined sides, so an edge stops it.
function cutGhosts(view) {
  const cut = view.cut;
  if (!cut.joined.length) return [];
  const rings = Math.max(1, Math.ceil(GHOST_MARGIN));
  const seen = new Set();
  let frontier = [];
  for (let v = 0; v < view.H; v++) {
    for (let u = 0; u < view.W; u++) {
      if (cellOnBoard(u, v, view)) { seen.add(u + ',' + v); frontier.push([u, v]); }
    }
  }
  const cells = [];
  for (let depth = 1; depth <= rings; depth++) {
    const next = [];
    for (const [u, v] of frontier) {
      for (const [du, dv] of HEX_STEPS) {
        const a = u + du, b = v + dv, key = a + ',' + b;
        if (seen.has(key)) continue;
        seen.add(key);
        const found = copyAt(a, b, view, cut.joined);
        if (!found) continue;
        cells.push({ square: found.square, u: a, v: b, depth: depth });
        next.push([a, b]);
      }
    }
    frontier = next;
  }
  // Drawn outermost first, so a nearer, stronger ring is painted over a farther one.
  return cells.sort((p, q) => q.depth - p.depth);
}

// A cell's centre on screen.
function cellCentre(u, v, cell, view) {
  const c = cellXY(u, v, cell, view);
  return [c[0] + cell / 2, c[1] + cell / 2];
}

// Each side of the hexagon as the cell edges it is made of, keyed by the side's name:
// every edge between a cell of the board and the cell beyond it, which lies in the copy
// across that side. `out` is the unit step from the board towards the copy. The side is
// a zigzag of cell edges, because that is where the seam is: a straight line would cut
// through cells on one side of it or the other.
function cutSides(cell, view) {
  const sides = {};
  const half = cell / (2 * Math.sqrt(3));
  for (let v = 0; v < view.H; v++) {
    for (let u = 0; u < view.W; u++) {
      if (!cellOnBoard(u, v, view)) continue;
      const here = cellCentre(u, v, cell, view);
      for (const [du, dv] of HEX_STEPS) {
        if (cellOnBoard(u + du, v + dv, view)) continue;
        const found = copyAt(u + du, v + dv, view, view.cut.copies);
        if (!found) continue;
        const there = cellCentre(u + du, v + dv, cell, view);
        const ox = (there[0] - here[0]) / cell, oy = (there[1] - here[1]) / cell;
        const mx = (here[0] + there[0]) / 2, my = (here[1] + there[1]) / 2;
        const side = sides[found.copy.side] || (sides[found.copy.side] = {
          copy: found.copy, edges: [] });
        side.edges.push({ x1: mx - oy * half, y1: my + ox * half,
                          x2: mx + oy * half, y2: my - ox * half,
                          mx: mx, my: my, ox: ox, oy: oy });
      }
    }
  }
  return sides;
}

// The same seams again where two ghost copies meet, keyed by pair: `barrierHtml`'s
// line on a rectangle runs on through the ghost rings, and a hexagon's does too. Two
// copies touch only where every pair is joined, and the side between them is the pair
// of the vector from one to the other -- the nw copy meets the n copy along an ne_sw
// seam -- so each edge is looked up in `copies` rather than worked out. A ghost beside
// the board lies on the board's own side, which `cutSides` already has.
function cutGhostSides(cell, view) {
  const ghosts = cutGhosts(view);
  if (!ghosts.length) return {};
  const found = new Map();
  for (const g of ghosts) {
    found.set(g.u + ',' + g.v, copyAt(g.u, g.v, view, view.cut.joined).copy.vector);
  }
  const half = cell / (2 * Math.sqrt(3));
  const pairs = {};
  for (const g of ghosts) {
    const mine = found.get(g.u + ',' + g.v);
    const here = cellCentre(g.u, g.v, cell, view);
    // Half the six steps, so each edge between two ghosts is met once.
    for (const [du, dv] of HEX_STEPS.slice(0, 3)) {
      const theirs = found.get((g.u + du) + ',' + (g.v + dv));
      if (!theirs || (theirs[0] === mine[0] && theirs[1] === mine[1])) continue;
      const across = view.cut.copies.find(copy =>
        copy.vector[0] === theirs[0] - mine[0] && copy.vector[1] === theirs[1] - mine[1]);
      if (!across) continue;
      const there = cellCentre(g.u + du, g.v + dv, cell, view);
      const ox = (there[0] - here[0]) / cell, oy = (there[1] - here[1]) / cell;
      const mx = (here[0] + there[0]) / 2, my = (here[1] + there[1]) / 2;
      (pairs[across.pair] || (pairs[across.pair] = { edges: [] })).edges.push({
        x1: mx - oy * half, y1: my + ox * half, x2: mx + oy * half, y2: my - ox * half });
    }
  }
  return pairs;
}

// The sides of each joined pair, sorted into the two kinds `seamAxes` sorts an axis
// into: a repeat, which every player may cross both ways and which is nowhere in
// particular, and a seam that is a fact of the game because somebody meets a rule there.
function cutSeams(rule) {
  const found = { repeat: [], barrier: [] };
  for (const pair of ['n_s', 'ne_sw', 'nw_se']) {
    const rows = (rule && rule[pair]) || [];
    if (!rows.some(row => row[0] || row[1])) continue;
    found[rows.every(row => row[0] && row[1]) ? 'repeat' : 'barrier'].push(pair);
  }
  return found;
}

// A side as one path through its edges.
function sidePath(side) {
  return side.edges.map(e => `M${e.x1} ${e.y1}L${e.x2} ${e.y2}`).join('');
}

// The repeat lines: a side every player crosses both ways, drawn thin like a
// rectangle's outline on a wrapping axis. On the full torus that is all six, which is
// the outline, and it is a repeat line rather than an edge because it is one repeat
// among many.
function cutOutlineHtml(cell, view, rule) {
  const repeat = cutSeams(rule).repeat;
  if (!repeat.length) return '';
  let out = '';
  for (const side of Object.values(cutSides(cell, view))) {
    if (!repeat.includes(side.copy.pair)) continue;
    out += `<path d="${sidePath(side)}" fill="none" stroke="var(--ink)"
      stroke-width="${cell * 0.055}" stroke-linecap="round" opacity="0.6"
      pointer-events="none"/>`;
  }
  return out;
}

// The sides as the rules draw them on screen, in the standard diagram's own terms: a
// turned board shows its top side at the bottom and still calls it the top.
const SIDE_WORDS = { n: 'top', s: 'bottom', ne: 'upper-right', sw: 'lower-left',
                     nw: 'upper-left', se: 'lower-right' };

// A seam somebody meets a rule at: `barrierHtml`'s line along each of the pair's two
// sides, with its markers, because a piece that may cross one way and not the other --
// or one player and not the other -- meets something there and nowhere else.
function cutBarrierHtml(cell, view, rule) {
  const barrier = cutSeams(rule).barrier;
  if (!barrier.length) return '';
  let out = '';
  // Where two ghost copies meet, the line alone and at full strength, as a rectangle's
  // seam runs on across its ghost rings bare: the markers stay on the board's sides.
  for (const [pair, side] of Object.entries(cutGhostSides(cell, view))) {
    if (!barrier.includes(pair)) continue;
    out += `<path d="${sidePath(side)}" fill="none" stroke="var(--seam)"
      stroke-width="${cell * SEAM_WIDTH}" stroke-linecap="round" opacity="0.95"
      pointer-events="none" data-ghost-seam="${pair}"/>`;
  }
  for (const [name, side] of Object.entries(cutSides(cell, view))) {
    const copy = side.copy;
    if (!barrier.includes(copy.pair)) continue;
    out += `<path d="${sidePath(side)}" fill="none" stroke="var(--seam)"
      stroke-width="${cell * SEAM_WIDTH}" stroke-linecap="round" opacity="0.95"
      pointer-events="none"/>`;
    out += sideMarksHtml(cell, side, rule[copy.pair], name, copy.high);
  }
  return out;
}

// What each player may do at this side, drawn on it: `seamMarksHtml`'s reading, along a
// side rather than an axis. `high` says whether this side is the pair's high one, which
// is the only thing that decides whether a permission points out of it or into it.
function sideMarksHtml(cell, side, rows, name, high) {
  const first = rows[0] || [false, false], second = rows[1] || [false, false];
  const same = first[0] === second[0] && first[1] === second[1];
  const players = same ? [[first, -1]] : [[first, 0], [second, 1]];
  const spots = same ? [0.5] : SEAM_MARKS;
  const n = side.edges.length;
  const ox = side.edges.reduce((s, e) => s + e.ox, 0) / n;
  const oy = side.edges.reduce((s, e) => s + e.oy, 0) / n;
  const len = Math.hypot(ox, oy) || 1;
  const out = [ox / len, oy / len];
  // Along the side, so the markers sit a third and two thirds of the way down it.
  const edges = side.edges.slice().sort((p, q) =>
    (p.mx * -out[1] + p.my * out[0]) - (q.mx * -out[1] + q.my * out[0]));
  let html = '';
  players.forEach(([allow, who], index) => {
    const at = edges[Math.round(spots[index] * (n - 1))];
    const leaves = high ? allow[1] : allow[0], enters = high ? allow[0] : allow[1];
    const fill = who === 0 ? 'var(--p0)' : who === 1 ? 'var(--p1)' : 'var(--seam)';
    html += `<g pointer-events="none"><title>${sideWords(leaves, enters, who, name)}</title>`;
    if (leaves) html += seamArrowAlong(at.mx, at.my, cell, out[0], out[1], fill);
    if (enters) html += seamArrowAlong(at.mx, at.my, cell, -out[0], -out[1], fill);
    html += `<circle cx="${at.mx}" cy="${at.my}" r="${cell * 0.14}" fill="${fill}"
      stroke="var(--seam)" stroke-width="${cell * 0.035}"/></g>`;
  });
  return html;
}

// `seamArrow`, pointing any way: `(dx, dy)` is the unit direction the piece goes.
function seamArrowAlong(cx, cy, cell, dx, dy, fill) {
  const point = (out, side) =>
    `${cx + (dx * out - dy * side) * cell},${cy + (dy * out + dx * side) * cell}`;
  const back = SEAM_ARROW - SEAM_HEAD;
  return `<polygon points="${point(SEAM_ARROW, 0)} ${point(back, 0.17)} ${
    point(back, -0.17)}" fill="${fill}" stroke="var(--seam)"
    stroke-width="${cell * 0.03}" stroke-linejoin="round"/>`;
}

// The marker's accessible name, naming the sides absolutely as `seamWords` names edges.
function sideWords(leaves, enters, who, name) {
  const opposite = { n: 's', s: 'n', ne: 'sw', sw: 'ne', nw: 'se', se: 'nw' }[name];
  const what = leaves && enters ? 'crosses this seam either way'
    : leaves ? 'leaves by the ' + SIDE_WORDS[name] + ' side'
    : enters ? 'leaves by the ' + SIDE_WORDS[opposite] + ' side'
    : 'may not cross this seam';
  const whose = who === 0 ? 'the first player'
    : who === 1 ? 'the second player' : 'either side';
  return whose + ' ' + what;
}

// The step to a square through each copy the mover may leave into, the shortest of
// them and the direct step, measured in cells. Ties go to the direct step, so a move
// that could be drawn either way is never drawn as a crossing.
function hexagonStep(from, to, W, vectors) {
  const du = to % W - from % W, dv = Math.floor(to / W) - Math.floor(from / W);
  const length = (a, b) => Math.max(Math.abs(a), Math.abs(b), Math.abs(a + b));
  let best = { files: du, ranks: dv, wrapped: false }, shortest = length(du, dv);
  for (const vector of vectors) {
    const a = du + vector[0], b = dv + vector[1];
    if (length(a, b) < shortest) {
      best = { files: a, ranks: b, wrapped: true };
      shortest = length(a, b);
    }
  }
  return best;
}

// How far apart two ranks of hexagons sit, in cells. Pointy-top hexagons one cell wide
// nest at this spacing, each rank half a cell right of the one below.
const HEX_ROW = Math.sqrt(3) / 2;

// What the drawing is cropped to: the outline, plus the margin the ghosts fade out
// across. Identical to the board's own extent where nothing wraps. A hex board is wider
// by the shear, and a hexagon's points reach past its box above and below.
function viewBox(view, cell) {
  if (view.cut) return cutViewBox(view, cell);
  if (view.hex) {
    const mx = view.files.margin * cell, my = view.ranks.margin * cell * HEX_ROW;
    const pad = cell * (1 / Math.sqrt(3) - 0.5);
    const w = (view.W + (view.H - 1) / 2) * cell, h = ((view.H - 1) * HEX_ROW + 1) * cell;
    return `${-mx} ${-my - pad} ${w + 2 * mx} ${h + 2 * my + 2 * pad}`;
  }
  const mx = view.files.margin * cell, my = view.ranks.margin * cell;
  return `${-mx} ${-my} ${view.W * cell + 2 * mx} ${view.H * cell + 2 * my}`;
}

// A hexagon's frame: the box its own cells fill, so at five cells a side the drawing is
// eleven cells wide rather than the rhombus's sixteen. Opened by the ghosts' margin where
// a copy is drawn, and otherwise by an arrowhead's reach where a seam carries markers
// that point out of the board and there is no ghost ring to hold them.
function cutViewBox(view, cell) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let v = 0; v < view.H; v++) {
    for (let u = 0; u < view.W; u++) {
      if (!cellOnBoard(u, v, view)) continue;
      const c = cellXY(u, v, cell, view);
      x0 = Math.min(x0, c[0]); y0 = Math.min(y0, c[1]);
      x1 = Math.max(x1, c[0] + cell); y1 = Math.max(y1, c[1] + cell);
    }
  }
  const pad = cell * (1 / Math.sqrt(3) - 0.5);
  const margin = view.cut.joined.length ? GHOST_MARGIN * cell
    : view.cut.marked ? cell * (SEAM_ARROW + 0.05) : 0;
  const mx = margin, my = margin + pad;
  return `${x0 - mx} ${y0 - my} ${x1 - x0 + 2 * mx} ${y1 - y0 + 2 * my}`;
}

// Where a square sits inside the outline, in squares from its bottom-left corner. This
// is the whole of what shifting the view does -- and it is a plain modulo, which is why
// a shift can be tested without drawing anything.
function squareCell(square, view) {
  const file = square % view.W, rank = Math.floor(square / view.W);
  return [wrapMod(file - view.files.shift, view.W),
          wrapMod(rank - view.ranks.shift, view.H)];
}

// The reverse: which square is drawn at a place in the grid. `u` and `v` run outside
// the outline for the ghost copies, and the answer is the same square again.
function cellSquare(u, v, view) {
  const file = wrapMod(u + view.files.shift, view.W);
  const rank = wrapMod(v + view.ranks.shift, view.H);
  return rank * view.W + file;
}

// A place in the grid as a pixel corner. The one place the board's orientation is
// decided, because the animation needs the same answer as the drawing: a flipped board
// that only half agreed with itself would send every piece off towards the corner it
// used to be in.
//
// On a hex lattice the corner is the box a hexagon is inscribed in: file `u` runs right,
// rank `v` runs up and half a cell right, so the six axial neighbours are the six cells
// that touch. Flipped is the half turn, which is the second player's own view under the
// `rotate` orientation a hex spec must use.
function cellXY(u, v, cell, view) {
  const xy = view.flipped
    ? [(view.W - 1 - u) * cell, v * cell]
    : [u * cell, (view.H - 1 - v) * cell];
  if (!view.hex) return xy;
  // Sheared from the square placement, so the flip above stays the only one: each row
  // above the bottom of the screen moves half a cell right and closes up to `HEX_ROW`.
  const above = ((view.H - 1) * cell - xy[1]) / cell;
  return [xy[0] + above * cell / 2, xy[1] * HEX_ROW];
}

// Where a square is drawn: the copy of it that lies inside the outline.
function squareXY(square, cell, view) {
  const place = squareCell(square, view);
  return cellXY(place[0], place[1], cell, view);
}

// The six corners of the pointy-top hexagon inscribed in a cell's box, clockwise from
// the top, shrunk about the centre by `inset` of a cell on each side.
function hexCorners(x, y, cell, inset) {
  const scale = 1 - 2 * (inset || 0);
  const cx = x + cell / 2, cy = y + cell / 2;
  const r = cell * scale / Math.sqrt(3), h = cell * scale / 2;
  return [[cx, cy - r], [cx + h, cy - r / 2], [cx + h, cy + r / 2],
          [cx, cy + r], [cx - h, cy + r / 2], [cx - h, cy - r / 2]];
}

function hexPath(x, y, cell, inset) {
  return 'M' + hexCorners(x, y, cell, inset).map(p => `${p[0]} ${p[1]}`).join('L') + 'Z';
}

// One cell's outline with `attrs` on it: a `<rect>` on a square lattice, the inscribed
// hexagon as a `<path>` on a hex one. Every renderer draws a cell-sized mark through this,
// so a tint, a wall and a selection all take the cell's own shape.
function cellShape(x, y, cell, view, attrs, inset) {
  if (view && view.hex) return `<path d="${hexPath(x, y, cell, inset)}" ${attrs}/>`;
  const k = cell * (inset || 0), side = cell - 2 * k;
  return `<rect x="${x + k}" y="${y + k}" width="${side}" height="${side}" ${attrs}/>`;
}

// A wall's hatch: corner to corner of a square, vertex to vertex of a hexagon.
function wallCross(x, y, cell, view) {
  if (!view || !view.hex) {
    return `M${x} ${y}L${x + cell} ${y + cell}M${x + cell} ${y}L${x} ${y + cell}`;
  }
  const c = hexCorners(x, y, cell, 0);
  return `M${c[5][0]} ${c[5][1]}L${c[2][0]} ${c[2][1]}M${c[1][0]} ${c[1][1]}L${c[4][0]} ${c[4][1]}`;
}

// A cell's colour. Two shades chequer a square board; a hex board needs three, because
// every hexagon touches six others, and `(file - rank) mod 3` differs across each edge.
function cellShade(file, rank, view) {
  if (!view || !view.hex) return (file + rank) % 2 === 0 ? 'var(--dark)' : 'var(--light)';
  return ['var(--light)', 'var(--dark)',
          'color-mix(in srgb, var(--light) 50%, var(--dark))'][wrapMod(file - rank, 3)];
}

// The copies of the board that surround the outline, one entry per square of them.
// `depth` is how many squares out the copy lies, which is what its paleness is read
// from; a corner takes the larger of its two distances, so the rings are concentric.
function ghostCells(view) {
  if (view.cut) return cutGhosts(view);
  const cells = [];
  const rf = view.files.rings, rr = view.ranks.rings;
  for (let v = view.H + rr - 1; v >= -rr; v--) {
    for (let u = -rf; u < view.W + rf; u++) {
      const du = u < 0 ? -u : (u >= view.W ? u - view.W + 1 : 0);
      const dv = v < 0 ? -v : (v >= view.H ? v - view.H + 1 : 0);
      const depth = Math.max(du, dv);
      if (!depth) continue;
      cells.push({ square: cellSquare(u, v, view), u: u, v: v, depth: depth });
    }
  }
  return cells;
}

// How pale a copy that far out is drawn. Paler than the board at once, and halving
// outwards, so the last ring is nearly gone by the time the frame cuts it off.
function ghostFade(depth) { return 0.34 * Math.pow(0.45, depth - 1); }

// One unit of repetition, under the frame of reference the reader has shifted to. Only
// a wrapping axis is marked: a cylinder's top and bottom are real edges, and boxing
// them in would call them a seam.
//
// The outline says where one repeat begins and ends in the frame currently on screen,
// and it moves with the shift. So it stands down on an axis whose seam is a fact --
// `seamAxes` -- where `barrierHtml` draws that seam where it really is. Two boundaries
// on one axis, one fixed and one movable, would leave a reader deciding which of them
// the rules meant, and only one of them is a rule.
function outlineHtml(cell, view, rule) {
  if (view.cut) return cutOutlineHtml(cell, view, rule);
  const w = view.W * cell, h = view.H * cell;
  const seams = seamAxes(rule);
  const line = (x1, y1, x2, y2) =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--ink)"
      stroke-width="${cell * 0.055}" stroke-linecap="square" opacity="0.6"
      pointer-events="none"/>`;
  let out = '';
  if (view.files.rings && !seams.files) out += line(0, 0, 0, h) + line(w, 0, w, h);
  if (view.ranks.rings && !seams.ranks) out += line(0, 0, w, 0) + line(0, h, w, h);
  return out;
}

// ------------------------------------------------- the seam, where it is a fact

// Whether an axis has a seam of its own. A seam every player may cross both ways is
// nowhere in particular: file `a` is beside file `h` exactly as `c` is beside `d`, and
// the outline is the only frame there is. Any other permission singles that one line
// out, because a piece that may cross it one way and not the other -- or not at all --
// meets something there and nowhere else on the axis. So the line is a fact of the
// game, and the board draws it rather than letting the reader infer it from a move that
// went the long way round.
function seamAxes(rule) {
  const pick = axis => {
    const rows = (rule && rule[axis]) || [];
    const wraps = rows.some(row => row[0] || row[1]);
    return wraps && !rows.every(row => row[0] && row[1]);
  };
  return { files: pick('files'), ranks: pick('ranks') };
}

// How thick the seam is drawn, how far its markers reach across it, and where along one
// repeat the two sides' markers sit. A seam both sides read the same way gets one
// marker, halfway.
const SEAM_WIDTH = 0.1;
const SEAM_ARROW = 0.47;
const SEAM_HEAD = 0.26;
const SEAM_MARKS = [0.33, 0.67];

// The seam, on every axis that has one. Drawn after the cells and over the outline, at
// full strength on a ghost ring as well as inside the repeat: a line that faded with
// the copy it crossed would say the seam stops there, and it does not.
function barrierHtml(cell, view, rule) {
  if (view.cut) return cutBarrierHtml(cell, view, rule);
  const seams = seamAxes(rule);
  let out = '';
  if (seams.files) out += seamHtml(cell, view, rule, false);
  if (seams.ranks) out += seamHtml(cell, view, rule, true);
  return out;
}

// One axis of it. The low edge is file `a` and rank 1, absolute from the standard
// diagram, so the seam is the line on the low side of that square wherever the shift
// has carried it -- and the shift is the whole of what moves it, which is what makes a
// reader shifting the view watch the board slide past a barrier that stays put.
//
// Every copy of the seam inside the frame is drawn, markers and all, because every copy
// is the same line under the same rule. A copy drawn bare beside a copy carrying arrows
// would invite a reader to look for the difference between them.
function seamHtml(cell, view, rule, down) {
  const rows = (rule && rule[down ? 'ranks' : 'files']) || [];
  const size = down ? view.H : view.W;
  const place = wrapMod(-(down ? view.ranks : view.files).shift, size);
  const at = down
    ? (view.flipped ? place * cell : (view.H - place) * cell)
    : (view.flipped ? (view.W - place) * cell : place * cell);
  const mx = view.files.margin * cell, my = view.ranks.margin * cell;
  const across = down ? [-mx, view.W * cell + mx] : [-my, view.H * cell + my];
  const along = down ? [-my, view.H * cell + my] : [-mx, view.W * cell + mx];
  let out = '';
  for (let lap = -3; lap <= 3; lap++) {
    const p = at + lap * size * cell;
    if (p < along[0] || p > along[1]) continue;
    const ends = down ? [across[0], p, across[1], p] : [p, across[0], p, across[1]];
    out += `<line x1="${ends[0]}" y1="${ends[1]}" x2="${ends[2]}" y2="${ends[3]}"
      stroke="var(--seam)" stroke-width="${cell * SEAM_WIDTH}" stroke-linecap="butt"
      opacity="0.95" pointer-events="none"/>`;
    out += seamMarksHtml(cell, view, rows, p, down);
  }
  return out;
}

// What each side may do at the line, drawn on the line. Both sides get a marker only
// where the two differ; where they read the seam the same way there is one marker in
// the seam's own colour, because a pair saying the same thing twice would only repeat
// the line. That is `goalMarks`'s reading of a square both sides win on, again.
function seamMarksHtml(cell, view, rows, at, down) {
  const first = rows[0] || [false, false], second = rows[1] || [false, false];
  const same = first[0] === second[0] && first[1] === second[1];
  const sides = same ? [[first, -1]] : [[first, 0], [second, 1]];
  const spots = same ? [0.5] : SEAM_MARKS;
  const span = (down ? view.W : view.H) * cell;
  let out = '';
  sides.forEach((side, index) => {
    const t = spots[index] * span;
    out += seamMark(down ? t : at, down ? at : t, cell, view, side[0], side[1], down);
  });
  return out;
}

// One side's reading: an arrowhead for each edge that side may leave by, pointing the
// way the piece goes, with a disc on the line between them. A bare disc is a side that
// may not cross here at all, which is a wall for them and nowhere else on the board.
//
// Filled in that side's own colour and edged in the seam's, never in `--ink`: `--ink`
// swaps ends between the themes, so a dark arrow edged in it reads as a pale one under
// a dark theme and the two sides then differ by a fill nobody looks at.
function seamMark(cx, cy, cell, view, allow, who, down) {
  // Which way a piece leaving by the low edge travels on screen -- off file `a` towards
  // file `h`, which a turned board draws the other way round. The same answer `stepXY`
  // gives for a step of one square.
  const low = down ? (view.flipped ? -1 : 1) : (view.flipped ? 1 : -1);
  const fill = who === 0 ? 'var(--p0)' : who === 1 ? 'var(--p1)' : 'var(--seam)';
  let out = `<g pointer-events="none"><title>${seamWords(allow, who, view, down)}</title>`;
  if (allow[0]) out += seamArrow(cx, cy, cell, low, fill, down);
  if (allow[1]) out += seamArrow(cx, cy, cell, -low, fill, down);
  out += `<circle cx="${cx}" cy="${cy}" r="${cell * 0.14}" fill="${fill}"
    stroke="var(--seam)" stroke-width="${cell * 0.035}"/>`;
  return out + '</g>';
}

function seamArrow(cx, cy, cell, sign, fill, down) {
  const point = (out, side) => down
    ? `${cx + side * cell},${cy + sign * out * cell}`
    : `${cx + sign * out * cell},${cy + side * cell}`;
  const back = SEAM_ARROW - SEAM_HEAD;
  return `<polygon points="${point(SEAM_ARROW, 0)} ${point(back, 0.17)} ${
    point(back, -0.17)}" fill="${fill}" stroke="var(--seam)"
    stroke-width="${cell * 0.03}" stroke-linejoin="round"/>`;
}

// The marker's accessible name. The arrow is the drawing and this is the sentence, in
// the compiled rule's own terms: a permission is *leaving by* an edge, named absolutely
// so that two identical strings can never mean two different seams.
function seamWords(allow, who, view, down) {
  const edge = high => down
    ? 'rank ' + (high ? view.H : 1)
    : 'the ' + String.fromCharCode(97 + (high ? view.W - 1 : 0)) + '-file';
  const what = allow[0] && allow[1] ? 'crosses this seam either way'
    : allow[0] ? 'leaves by ' + edge(false)
    : allow[1] ? 'leaves by ' + edge(true)
    : 'may not cross this seam';
  const whose = who === 0 ? 'the first player'
    : who === 1 ? 'the second player' : 'either side';
  return whose + ' ' + what;
}

// What the engine calls a square -- `chr('a' + file)` and `rank + 1`, from
// `engine._square_name`. One place, because the labels on the board and the sentence
// saying which repeat is on screen have to agree with the move list.
function squareName(file, rank) { return String.fromCharCode(97 + file) + (rank + 1); }

// Which of a square's sides carry a name: the bottom edge of the frame takes a file's
// letter and the left edge a rank's number, wherever on screen the frame's first rank and
// file are after a flip or a shift. Where a square sits is its place in the grid, so a
// sheared hex board's edges are found the same way as a square board's.
//
// A hexagon (schema 16): a file's letter on its lowest cell on screen and a rank's
// number on its leftmost, which is the cell with no cell of the board below it or to
// its left. Letters a to f and ranks 1 to 6 then run up the lower-left side, letters f
// to k along the bottom and ranks 6 to 11 up the upper-left side.
function labelSides(square, view) {
  const file = square % view.W, rank = Math.floor(square / view.W);
  if (view.cut) {
    const way = view.flipped ? -1 : 1;
    return { bottom: !cellOnBoard(file, rank - way, view),
             left: !cellOnBoard(file - way, rank, view) };
  }
  const place = squareCell(square, view);
  return { bottom: place[1] === (view.flipped ? view.H - 1 : 0),
           left: place[0] === (view.flipped ? view.W - 1 : 0) };
}

// Whether the names can stand outside the board, in a margin of their own where no piece
// and no mark covers them (plan-visual.md §3.3): wherever the drawing ends at the board's
// own edge. A wrapping board's margin is its ghost copies and a hexagon with a seam has
// its seam's arrows there, so those keep their names on the naming frame's own squares.
function labelsOutside(view) {
  if (view.files.margin || view.ranks.margin) return false;
  return !(view.cut && (view.cut.joined.length || view.cut.marked));
}

// The names for the margin, one per edge that carries one: the name, the middle of the
// cell edge it belongs beside (in the drawing's units), and the unit step outwards from
// the board across that edge, along which the page sets it off the board by a fixed
// distance on screen. The edge is the one between the square and the place beside it
// that is off the board, below it for a file and to its left for a rank, so a hexagon's
// names run along its slanted sides as its cells do.
function edgeLabels(cell, view) {
  const way = view.flipped ? -1 : 1;
  const out = [];
  for (let square = 0; square < view.W * view.H; square++) {
    if (cellAbsent(square, view)) continue;
    const sides = labelSides(square, view);
    if (!sides.bottom && !sides.left) continue;
    const place = squareCell(square, view);
    const name = squareName(square % view.W, Math.floor(square / view.W));
    const at = cellCentre(place[0], place[1], cell, view);
    const beside = (du, dv, text) => {
      const next = cellCentre(place[0] + du, place[1] + dv, cell, view);
      const dx = next[0] - at[0], dy = next[1] - at[1], length = Math.hypot(dx, dy) || 1;
      out.push({ text: text, x: (at[0] + next[0]) / 2, y: (at[1] + next[1]) / 2,
                 dx: dx / length, dy: dy / length });
    };
    if (sides.bottom) beside(0, -way, name.slice(0, 1));
    if (sides.left) beside(-way, 0, name.slice(1));
  }
  return out;
}

// The name of a square, on the two edges of the outline where it fits.
//
// The same names the engine uses in the move list -- `chr('a' + file)` and `rank + 1`,
// from `engine._square_name` -- so "Nb1-c3" in the panel can be found on the board
// without counting squares from a corner. That agreement is the whole point, which is
// also why the label is the square's *own* name and not its position on screen: turning
// the board around, or shifting it across the seam, must not renumber it.
//
// So the labels move with the shift. A view shifted three files along reads
// `d e f g h a b c` across the bottom, which is what says the board has been shifted
// rather than renamed. Ghost copies carry none: a label belongs to the frame of
// reference, and the outline is where that frame is.
function coordsHtml(square, x, y, cell, view) {
  const file = square % view.W, rank = Math.floor(square / view.W);
  const sides = labelSides(square, view);
  const bottom = sides.bottom, left = sides.left;
  if (!bottom && !left) return '';
  // Against the square it sits on, whichever that is: a fixed colour is invisible on
  // half of any chequerboard. The halo is the square's own colour, painted under the
  // glyph, because a corner is only *usually* empty -- a piece standing on a1 would
  // otherwise swallow the letter under it.
  const dark = cellShade(file, rank, view) === 'var(--dark)';
  const ink = dark ? 'var(--light)' : 'var(--dark)';
  const halo = dark ? 'var(--dark)' : 'var(--light)';
  const style = `font-size="${(cell * 0.24).toFixed(2)}" fill="${ink}" stroke="${halo}"
    stroke-width="${(cell * 0.05).toFixed(3)}" paint-order="stroke" pointer-events="none"`;
  const name = squareName(file, rank);
  let out = '';
  if (bottom) {
    out += `<text class="co" x="${(x + cell * 0.94).toFixed(2)}" y="${
      (y + cell * 0.95).toFixed(2)}" ${style}
      text-anchor="end">${name.slice(0, 1)}</text>`;
  }
  if (left) {
    out += `<text class="co" x="${(x + cell * 0.07).toFixed(2)}" y="${
      (y + cell * 0.26).toFixed(2)}" ${style}>${name.slice(1)}</text>`;
  }
  return out;
}

// --------------------------------------------- squares the rules single out

// What each kind of mark looks like: a colour of its own, a shape, and the words the
// board's legend line says for it. A ring (the default shape) also has how far inside the
// square it sits and which corner its beads hang on; the inset and the corner belong to
// the kind and every coordinate below is derived from them, so two rings on one square
// nest rather than land on top of each other and their beads sit at opposite ends of it.
//
// The shapes say what kind of thing the mark is (plan-visual.md §2, S3). A **ring** is a
// target: a square a piece is trying to reach. A **hatch** is ground: a region of the
// board that a rule covers, drawn under the piece and never in the shape of a pick --
// the promotion zone was a ring and every first-time reader read it as something else
// (castling rights, *special*, *selected*; the first-visit study's F1). A **glow** is
// *this game, now*: a tint of the whole square, which is how the board already says
// last move and selection.
//
// `name` and `label` are the legend's: what the mark looks like and what it means, so
// the line under the board is generated from the row that draws the mark and a kind
// cannot be drawn without words for it.
const MARK_KINDS = {
  // The royal piece the engine says is attacked (`inCheck`), on the square it stands
  // on. First, so it is the bottom layer: a tint under the rings and the piece.
  check: { colour: 'var(--alarm)', shape: 'glow', name: 'red square',
           label: 'a royal piece under attack (check)' },
  goal: { colour: 'var(--goal)', inset: 0.085, corner: 'top', name: 'teal ring',
          label: 'a square that wins the game' },
  // A promotion zone: hatched ground, with no beads. Whose zone it is goes in the legend
  // line (`zoneWords`), because a bead on a square was read as a dot, a pick or a piece.
  zone: { colour: 'var(--zone)', inset: 0, corner: 'bottom', shape: 'hatch',
          name: 'rose hatching', label: 'a square a piece promotes on' },
  // A seam a piece crosses one way only, or one side only. Drawn on the board's edges by
  // `barrierHtml`, never on a square, so `squareMarks` adds none; the row is here so the
  // legend's entry and its key come from the same place as every other kind's.
  seam: { colour: 'var(--seam)', shape: 'edge', name: 'blue line',
          label: 'an edge only one side may cross, or may cross one way' },
  // Where a wild piece steps after the next move (Q660): an arrow between two squares,
  // drawn by `comingHtml` off the core's answer (`engine.coming.coming_phase`), never on one
  // square, so `squareMarks` adds none. A wild piece walks a fixed route and nothing about
  // it is drawn at random; the coming move can still block it, clear its way or take it.
  // `ink` is the legend's name in words: the arrow's violet is dark enough for the wood in
  // both themes, so on a dark page its name takes a lighter violet of its own.
  coming: { colour: 'var(--coming)', shape: 'arrow', name: 'dotted arrow',
            ink: 'var(--coming-ink)',
            label: 'where a wild piece steps after the next move' },
  // And where it steps after the move after that (Q764): a wild piece walks twice before
  // the player to move moves again, after that player's move and after the reply. The
  // same violet at the same opacity, so it still reads 3 : 1 on both woods, and fainter
  // by carrying less ink: smaller dots set further apart and a smaller head
  // (`COMING_FAINT`). Its own row, so the legend names it in its own words.
  comingThen: { colour: 'var(--coming)', shape: 'arrow', faint: true,
                name: 'faint dotted arrow', ink: 'var(--coming-ink)',
                label: 'where it steps after the move after that' },
  // Every wild piece's walk, in order (Q852): a path of arrows from square to square, one
  // hue per walk (`--walk-N`, cased in `--walk-edge`), dashed where a step jumps to a square
  // that does not touch. Drawn by a still board (`walksHtml`, off the payload's `walks`),
  // never on one square, so `squareMarks` adds none; the row is the legend's.
  walk: { colour: 'var(--walk-1)', shape: 'walk', name: 'coloured arrows', ink: 'var(--ink)',
          label: "each wild piece's walk, in order, a colour per walk; a dashed arrow jumps to a square that does not touch" },
  // A wall (`obstacles`): a square drawn dark and crossed, which no piece may stand on.
  // Drawn by the board's own cell loop, never as a mark, so `squareMarks` adds none; the
  // row is the legend's, named where the board has a wall (Q772b). Its words are in ink,
  // the wall's own colour being nearly the dark page's.
  wall: { colour: 'var(--wall)', shape: 'wall', name: 'dark crossed square', ink: 'var(--ink)',
          label: 'a wall: no piece may stand on it' },
  // A portal end (schema 19): drawn as a piece is, by `portalGlyph`, so `squareMarks`
  // adds none either; the row is the legend's, named where the game has portals. `ink`
  // is the words' own colour on the page, the pair's colour being drawn for the wood.
  // Its (i) (`gloss`, `GLOSS.portalKey`) draws four ends with their sentences (Q838c).
  portal: { colour: 'var(--pair-1)', shape: 'portal', name: 'portal',
            ink: 'var(--ink)', gloss: 'portalKey',
            label: "pieces of the arms' colours go in, and come out of the same-coloured portal its beak points at" },
  // A fold portal (schema 22, Q843d, `docs/archive/plan-portals-2.md` §5.3): a pool lying in the
  // board, drawn by the board's cell loop (`foldGlyph`), never a mark; the row is the
  // legend's, named where the game has a fold. Its colour is the pool's bright centre, which
  // only a fold paints. Its (i) (`GLOSS.foldKey`) draws a pool with a piece in it, and a pair
  // whose beaks and map arrows say where a piece comes out (§5.3R).
  fold: { colour: 'var(--fold-light)', shape: 'fold', name: 'fold portal',
          ink: 'var(--ink)', gloss: 'foldKey',
          label: 'a piece that stops here may stay, or come out of the end its beak points at' },
  // Schema 26 (Q918c, `docs/reference/rounds.md` §2.7): two pieces of a game in rounds that
  // could only take each other by a capture-only move, after one void round between them
  // (`voidRun`, dashed, a "1" at its middle) and after two in a row (`barred`, solid, a bar
  // across it), when that pair of moves is not offered until either square changes. Drawn
  // between the two squares from the core's state (`standoffLine`, the round's `voidRun`
  // and `bars`), never on a square; the rows are the legend's. Each in a colour of its own,
  // neither the held move's `--held` nor the other's (Q940): a kind is told by its colour,
  // by a reader and by the look check, which read a held move as both kinds while they
  // shared it.
  voidRun: { colour: 'var(--standoff-run)', shape: 'link', name: 'dashed link', ink: 'var(--ink)',
             label: 'one void round between these two: a second in a row bars those moves' },
  barred: { colour: 'var(--standoff-bar)', shape: 'link', barred: true, name: 'barred link',
            ink: 'var(--ink)',
            label: 'two void rounds in a row: neither may take the other that way until one of the squares changes' },
  // Schema 27 (Q942b, `docs/archive/plan-round-chase.md` §3 R2): two pieces that swapped squares
  // and met, where `swap: "collision"` makes a swap a meeting. Drawn between the two
  // squares, a burst at the middle where they met (`clashMark`), off the core's event in
  // the last round; the row is the legend's.
  clash: { colour: 'var(--clash)', shape: 'clash', name: 'burst', ink: 'var(--ink)',
           label: 'two pieces that swapped squares and met half-way, as the rules make a swap a meeting' },
  // Schema 27 (§3 R1, `rest`): the piece each player moved last round, which rests this one.
  // A pause sign on the square's top corner (`restBadge`), off the core's rested squares.
  rested: { colour: 'var(--rest)', shape: 'rest', name: 'pause sign', ink: 'var(--ink)',
            label: 'the piece moved last round: it rests and may not move this round, unless it is all its side can move' },
  // Schema 28 (Q943b, `docs/plan-speed.md` §4.5): the order the last round's two moves were
  // made in, where the two pieces differed in speed: a numbered disc on the bottom corner of
  // each move's square (`orderBadge`), off the core's events (`roundOrder`).
  order: { colour: 'var(--order)', shape: 'order', name: 'numbered disc', ink: 'var(--ink)',
           label: "the order the last round's two moves were made in: between pieces of different speeds the faster moves first" },
  // Schema 27 (Q935c, `docs/plan-spy.md` §3.6): the move announced in a spy game's announced
  // round, shown to both players before the answer. A solid arrow and a ring on the square
  // it goes to (`announceLine`, `announcedHtml`), in a teal of its own: never the held
  // move's dashed pine, so a move shown and a move held unsent are two marks on one board.
  announced: { colour: 'var(--announced)', shape: 'announce', name: 'solid arrow',
               ink: 'var(--ink)',
               label: 'the move announced this round: it is made, whatever the other player answers' },
  // Schema 30 (Q999c, `docs/plan-passive.md` §4.6): a piece under a line of sight's effect,
  // off the core's effect bits on the position on screen (`Engine.effects`): a small badge
  // on the square's left corners, clear of the rested piece's sign and the order disc. A
  // frozen piece's snowflake and a disarmed piece's broken blade share the top corner (a
  // frozen piece cannot capture either, so its snowflake says both); a shielded piece's
  // shield is on the bottom one. Each a shape and a colour of its own, on `held-halo`.
  frozen: { colour: 'var(--frozen)', shape: 'frozen', name: 'snowflake', ink: 'var(--ink)',
            label: 'frozen: it cannot move, and gives no check' },
  disarmed: { colour: 'var(--disarmed)', shape: 'disarmed', name: 'broken blade',
              ink: 'var(--ink)', label: 'disarmed: it may move but cannot capture' },
  shielded: { colour: 'var(--shielded)', shape: 'shielded', name: 'shield', ink: 'var(--ink)',
              label: 'shielded: it cannot be captured' },
  // What a piece sees (schema 30), tinted over the squares while the pointer is on it or it
  // was tapped (`sightHtml`), off the core's `sight_of`; the row is the legend's, named in
  // every game with a line of sight.
  seen: { colour: 'var(--frozen)', shape: 'seen', name: 'tinted squares', ink: 'var(--ink)',
          label: 'what a piece sees: point at it, or tap it, and what stands there is under its effect' },
};

// Schema 30 (Q999c): the effect bits the core reports for a piece (`Engine.effects`), and
// the mark kind each draws. Freeze contains disarm, so a piece both frozen and disarmed
// wears the snowflake alone.
const SIGHT_FREEZE = 1, SIGHT_DISARM = 2, SIGHT_SHIELD = 4;

function effectKinds(bits) {
  const kinds = [];
  if (bits & SIGHT_FREEZE) kinds.push('frozen');
  else if (bits & SIGHT_DISARM) kinds.push('disarmed');
  if (bits & SIGHT_SHIELD) kinds.push('shielded');
  return kinds;
}

// The one effect a seen square is tinted by, where a piece's sights put more than one on it:
// freeze first, then disarm, then shield, as the badges read.
function effectTint(bits) {
  return bits & SIGHT_FREEZE ? 'frozen' : bits & SIGHT_DISARM ? 'disarmed' : 'shielded';
}

// A frozen piece's badge on the top-left corner of the square at `(x, y)`: a disc in
// `--frozen` cased in `--held-halo`, with a six-armed snowflake in the casing's colour.
function frozenBadge(x, y, cell, player) {
  const f = n => n.toFixed(3);
  const cx = x + cell * 0.2, cy = y + cell * 0.2, r = cell * 0.15, arm = cell * 0.105;
  let arms = '';
  for (let k = 0; k < 3; k++) {
    const a = Math.PI * k / 3 + Math.PI / 2;
    const dx = arm * Math.cos(a), dy = arm * Math.sin(a);
    arms += `<line x1="${f(cx - dx)}" y1="${f(cy - dy)}" x2="${f(cx + dx)}" y2="${f(cy + dy)}"/>`;
  }
  return `<g data-mark="frozen"${player === undefined ? '' : ` data-player="${player}"`}
      pointer-events="none"><circle cx="${f(cx)}" cy="${f(cy)}" r="${f(r)}" fill="var(--frozen)"
      stroke="var(--held-halo)" stroke-width="${f(cell * 0.03)}"/><g stroke="var(--held-halo)"
      stroke-width="${f(cell * 0.03)}" stroke-linecap="round">${arms}</g></g>`;
}

// A disarmed piece's badge, on the same corner: a disc in `--disarmed` cased in `--held-halo`,
// with a blade broken across its middle (two strokes either side of a gap) in the casing's
// colour.
function disarmedBadge(x, y, cell, player) {
  const f = n => n.toFixed(3);
  const cx = x + cell * 0.2, cy = y + cell * 0.2, r = cell * 0.15;
  const a = cell * 0.095, gap = cell * 0.025, off = cell * 0.025;
  // Along the diagonal from bottom left to top right, the top half nudged off the line.
  const half = (s, shift) => `<line x1="${f(cx + s * gap + shift)}" y1="${f(cy - s * gap + shift)}"
      x2="${f(cx + s * a + shift)}" y2="${f(cy - s * a + shift)}"/>`;
  return `<g data-mark="disarmed"${player === undefined ? '' : ` data-player="${player}"`}
      pointer-events="none"><circle cx="${f(cx)}" cy="${f(cy)}" r="${f(r)}" fill="var(--disarmed)"
      stroke="var(--held-halo)" stroke-width="${f(cell * 0.03)}"/><g stroke="var(--held-halo)"
      stroke-width="${f(cell * 0.035)}" stroke-linecap="round">${half(-1, 0)}${half(1, -off)}</g></g>`;
}

// A shielded piece's badge on the bottom-left corner: a shield's outline (flat top, pointed
// foot) filled `--shielded` and cased in `--held-halo`.
function shieldedBadge(x, y, cell, player) {
  const f = n => n.toFixed(3);
  const cx = x + cell * 0.2, top = y + cell * 0.66, w = cell * 0.13, h = cell * 0.27;
  const d = `M${f(cx - w)} ${f(top)}L${f(cx + w)} ${f(top)}L${f(cx + w)} ${f(top + h * 0.45)}Q${
    f(cx + w)} ${f(top + h * 0.8)} ${f(cx)} ${f(top + h)}Q${f(cx - w)} ${f(top + h * 0.8)} ${
    f(cx - w)} ${f(top + h * 0.45)}Z`;
  return `<g data-mark="shielded"${player === undefined ? '' : ` data-player="${player}"`}
      pointer-events="none"><path d="${d}" fill="var(--shielded)" stroke="var(--held-halo)"
      stroke-width="${f(cell * 0.03)}" stroke-linejoin="round"/></g>`;
}

// The badges of one piece's effects (`effectKinds`), on its square at `(x, y)`.
function effectBadges(bits, x, y, cell, player) {
  const draw = { frozen: frozenBadge, disarmed: disarmedBadge, shielded: shieldedBadge };
  return effectKinds(bits).map(kind => draw[kind](x, y, cell, player)).join('');
}

// One square a piece sees (schema 30), tinted in its effect's colour with an inset edge, for
// the hover; `source` rings the seeing piece's own square instead.
function seenSquare(x, y, cell, bits, view, source) {
  const kind = MARK_KINDS[effectTint(bits)];
  if (source) {
    return cellShape(x, y, cell, view, `fill="none" stroke="${kind.colour}"
        stroke-width="${(cell * 0.06).toFixed(3)}" data-mark="seen" data-source="1"
        pointer-events="none"`, 0.06);
  }
  return cellShape(x, y, cell, view, `fill="${kind.colour}" fill-opacity="${SEEN_OPACITY}"
      stroke="${kind.colour}" stroke-width="${(cell * 0.04).toFixed(3)}" stroke-opacity="0.85"
      data-mark="seen" data-effect="${effectTint(bits)}" pointer-events="none"`, 0.02);
}

// How strongly a seen square is tinted (`tokens.BOARD_MARKS` reads it at this opacity).
const SEEN_OPACITY = 0.3;

// An announced move's arrow (schema 27, Q935c), from `(x1, y1)` to its tip at `(x2, y2)`: a
// solid shaft and a filled head in `--announced`, cased in `--held-halo` as the held move's
// arrow is, so it reads on both woods; solid where the held move's is dashed.
function announceLine(x1, y1, x2, y2, cell) {
  const f = n => n.toFixed(3);
  const length = Math.hypot(x2 - x1, y2 - y1);
  if (!length) return '';
  const ux = (x2 - x1) / length, uy = (y2 - y1) / length;
  const head = Math.min(cell * 0.26, length * 0.6);
  const back = [x2 - ux * head, y2 - uy * head];
  const wing = [-uy * head * 0.7, ux * head * 0.7];
  const tip = `M${f(back[0] + wing[0])} ${f(back[1] + wing[1])}L${f(x2)} ${f(y2)}L${
    f(back[0] - wing[0])} ${f(back[1] - wing[1])}Z`;
  const line = `x1="${f(x1)}" y1="${f(y1)}" x2="${f(back[0])}" y2="${f(back[1])}"`;
  return `<g data-mark="announced" pointer-events="none" stroke-linecap="round"
      stroke-linejoin="round"><line ${line} stroke="var(--held-halo)"
      stroke-width="${f(cell * 0.17)}" opacity="0.9"/><path d="${tip}" fill="var(--held-halo)"
      stroke="var(--held-halo)" stroke-width="${f(cell * 0.1)}" opacity="0.9"/><line ${line}
      stroke="var(--announced)" stroke-width="${f(cell * 0.09)}"/><path d="${tip}"
      fill="var(--announced)" stroke="var(--announced)" stroke-width="${f(cell * 0.03)}"/></g>`;
}

// A swap's meeting between the middles of two squares (schema 27): a short link in
// `--clash`, cased in `--held-halo` as a standoff's is, with an eight-pointed burst at its
// middle, where the two pieces met.
function clashMark(x1, y1, x2, y2, cell) {
  const f = n => n.toFixed(3);
  const length = Math.hypot(x2 - x1, y2 - y1);
  if (!length) return '';
  const ux = (x2 - x1) / length, uy = (y2 - y1) / length;
  const trim = Math.min(cell * 0.3, length / 3);
  const line = `x1="${f(x1 + ux * trim)}" y1="${f(y1 + uy * trim)}" x2="${
    f(x2 - ux * trim)}" y2="${f(y2 - uy * trim)}"`;
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const points = [];
  for (let k = 0; k < 16; k++) {
    const r = cell * (k % 2 ? 0.1 : 0.22), a = Math.PI * k / 8 + Math.PI / 16;
    points.push(`${f(mx + r * Math.cos(a))},${f(my + r * Math.sin(a))}`);
  }
  return `<g data-mark="clash" pointer-events="none" stroke-linecap="round"><line ${line}
      stroke="var(--held-halo)" stroke-width="${f(cell * 0.13)}" opacity="0.85"/><line ${line}
      stroke="var(--clash)" stroke-width="${f(cell * 0.06)}"/><polygon points="${
      points.join(' ')}" fill="var(--clash)" stroke="var(--held-halo)"
      stroke-width="${f(cell * 0.035)}" stroke-linejoin="round"/></g>`;
}

// A rested piece's pause sign (schema 27) on the top corner of the square at `(x, y)`, clear
// of the piece's head: a disc in `--rest` cased in `--held-halo`, two bars in the casing's
// colour. `player` is whose piece rests, for a reader of the drawing.
function restBadge(x, y, cell, player) {
  const f = n => n.toFixed(3);
  const cx = x + cell * 0.8, cy = y + cell * 0.2, r = cell * 0.15;
  const w = cell * 0.045, h = cell * 0.13;
  const bar = dx => `<rect x="${f(cx + dx - w / 2)}" y="${f(cy - h / 2)}" width="${f(w)}"
      height="${f(h)}" rx="${f(w / 3)}" fill="var(--held-halo)"/>`;
  return `<g data-mark="rested"${player === undefined ? '' : ` data-player="${player}"`}
      pointer-events="none"><circle cx="${f(cx)}" cy="${f(cy)}" r="${f(r)}" fill="var(--rest)"
      stroke="var(--held-halo)" stroke-width="${f(cell * 0.03)}"/>${bar(-cell * 0.04)}${
      bar(cell * 0.04)}</g>`;
}

// A move's place in an ordered round (schema 28, Q943b), `n` 1 or 2, on the bottom corner of
// the square at `(x, y)`, clear of the rested piece's sign on the top one: a disc in
// `--held-halo` ringed and numbered in `--order`, dark on light in both themes. `right` puts it on
// the other bottom corner, for the second of two moves that ended on one square.
function orderBadge(x, y, cell, n, player, right) {
  const f = n => n.toFixed(3);
  const cx = x + cell * (right ? 0.8 : 0.2), cy = y + cell * 0.8, r = cell * 0.16;
  return `<g data-mark="order" data-order="${n}"${player === undefined ? '' : ` data-player="${
      player}"`} pointer-events="none"><circle cx="${f(cx)}" cy="${f(cy)}" r="${f(r)}"
      fill="var(--held-halo)" stroke="var(--order)" stroke-width="${f(cell * 0.025)}"/><text
      x="${f(cx)}" y="${f(cy)}" fill="var(--order)" font-size="${f(cell * 0.22)}" font-weight="700"
      text-anchor="middle" dominant-baseline="central">${n}</text></g>`;
}

// A standoff's link between the middles of two squares, `(x1, y1)` and `(x2, y2)`, in the
// look of the mark kind `kind` (`voidRun` or `barred`): a line in the kind's own colour cased
// in `--held-halo`, so it reads on both woods as the held move's arrow does; dashed with a "1" for a run of
// one void round, solid with a bar across its middle once the pair is barred.
function standoffLine(x1, y1, x2, y2, cell, kind) {
  const f = n => n.toFixed(3);
  const look = MARK_KINDS[kind] || MARK_KINDS.voidRun;
  const length = Math.hypot(x2 - x1, y2 - y1);
  if (!length) return '';
  const ux = (x2 - x1) / length, uy = (y2 - y1) / length;
  const trim = Math.min(cell * 0.3, length / 3);
  const ax = x1 + ux * trim, ay = y1 + uy * trim, bx = x2 - ux * trim, by = y2 - uy * trim;
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const line = `x1="${f(ax)}" y1="${f(ay)}" x2="${f(bx)}" y2="${f(by)}"`;
  const dash = look.barred ? '' : ` stroke-dasharray="${f(cell * 0.14)} ${f(cell * 0.1)}"`;
  let out = `<line ${line} stroke="var(--held-halo)" stroke-width="${f(cell * 0.13)}"
      opacity="0.85"/><line ${line} stroke="${look.colour}" stroke-width="${f(cell * 0.06)}"${
      dash}/>`;
  if (look.barred) {
    const wx = -uy * cell * 0.17, wy = ux * cell * 0.17;
    const bar = `x1="${f(mx + wx)}" y1="${f(my + wy)}" x2="${f(mx - wx)}" y2="${f(my - wy)}"`;
    out += `<line ${bar} stroke="var(--held-halo)" stroke-width="${f(cell * 0.14)}"
        stroke-linecap="round"/><line ${bar} stroke="${look.colour}"
        stroke-width="${f(cell * 0.07)}" stroke-linecap="round"/>`;
  } else {
    out += `<circle cx="${f(mx)}" cy="${f(my)}" r="${f(cell * 0.13)}" fill="var(--held-halo)"
        stroke="${look.colour}" stroke-width="${f(cell * 0.03)}"/><text x="${f(mx)}"
        y="${f(my + cell * 0.055)}" text-anchor="middle" font-size="${f(cell * 0.16)}"
        font-weight="700" fill="${look.colour}">1</text>`;
  }
  return `<g data-mark="${kind}" pointer-events="none" stroke-linecap="round">${out}</g>`;
}

// The coming step's arrow, in fractions of a cell: it leaves the wild piece's square
// `COMING_FROM` out from the middle, clear of the piece's disc, and stops `COMING_SHORT`
// short of the middle of the square it is going to, clear of a legal-move dot (r .13
// plus its stroke). `COMING_OPACITY` is what `tokens.BOARD_MARKS` reads it at.
const COMING_FROM = 0.42;
const COMING_SHORT = 0.2;
const COMING_HEAD = 0.15;
const COMING_WIDTH = 0.07;
const COMING_OPACITY = 0.85;
// The step after that one (Q764), drawn fainter by drawing less of it rather than by
// fading its colour, which would take it under 3 : 1 on the dark wood: dots `width`
// across set `gap` widths apart, and a head `head` of a cell. It leaves its square only
// `from` out from the middle: that square is where the first arrow points, the wild
// piece is not on it yet, and starting near the first arrow's tip makes the two read as
// one route, there and then on, rather than two arrows leaving one corner.
const COMING_FAINT = { width: 0.05, gap: 3.4, head: 0.12, from: 0.24 };
// How far a step and the step straight back are set apart, each to its own right, so
// a wild piece on a two-square route shows two arrows side by side rather than one
// drawn over the other.
const COMING_APART = 0.13;

// One arrow from a cell's middle `(cx, cy)` along the step `(dx, dy)`, set `apart` of a
// cell to its own right, in the look of the mark kind `kind`.
function comingArrow(cx, cy, dx, dy, cell, kind, apart) {
  const length = Math.hypot(dx, dy);
  const faint = MARK_KINDS[kind || 'coming'].faint;
  const from = cell * (faint ? COMING_FAINT.from : COMING_FROM);
  const to = length - cell * COMING_SHORT;
  if (!length || to <= from) return '';
  const ux = dx / length, uy = dy / length;
  const side = cell * (apart || 0), sx = -uy * side, sy = ux * side;
  return comingLine(cx + ux * from + sx, cy + uy * from + sy,
                    cx + ux * to + sx, cy + uy * to + sy, cell, kind);
}

// The arrow itself, `(x1, y1)` to its tip at `(x2, y2)`: a shaft of round dots and a
// chevron, both in the kind's colour, sized off `cell`. `kind` is `coming` (the default)
// or `comingThen`, the fainter step after it.
function comingLine(x1, y1, x2, y2, cell, kind) {
  const length = Math.hypot(x2 - x1, y2 - y1);
  if (!length) return '';
  const name = kind || 'coming';
  const faint = MARK_KINDS[name].faint ? COMING_FAINT : null;
  const ux = (x2 - x1) / length, uy = (y2 - y1) / length;
  const head = cell * (faint ? faint.head : COMING_HEAD);
  const width = cell * (faint ? faint.width : COMING_WIDTH);
  const gap = faint ? faint.gap : 2.1;
  const back = [x2 - ux * head, y2 - uy * head];
  const left = [back[0] - uy * head * 0.8, back[1] + ux * head * 0.8];
  const right = [back[0] + uy * head * 0.8, back[1] - ux * head * 0.8];
  const f = n => n.toFixed(3);
  return `<g data-mark="${name}" fill="none" stroke="${MARK_KINDS[name].colour}"
      stroke-width="${f(width)}" stroke-linecap="round" stroke-linejoin="round"
      opacity="${COMING_OPACITY}" pointer-events="none"><line x1="${f(x1)}" y1="${f(y1)}"
      x2="${f(back[0])}" y2="${f(back[1])}" stroke-dasharray="0.001 ${f(width * gap)}"/><path
      d="M${f(left[0])} ${f(left[1])}L${f(x2)} ${f(y2)}L${f(right[0])} ${f(right[1])}"/></g>`;
}

// Every wild piece's coming steps on one drawing: `coming` is the core's
// `[{from, to, eaten}]` for the phase after the next move on the position on screen, and
// `then` the phase after the move after that (Q764), drawn fainter and under it. A wild
// piece belongs to neither seat, so its step crosses whatever seams the board has, as its
// walk does (`animatePhase`). Nothing is worked out here but where on screen the squares
// are, and which drawn steps are one another's way back, to set those side by side.
function comingHtml(coming, cell, W, H, view, rule, then) {
  coming = coming || [];
  then = then || [];
  if (!coming.length && !then.length) return '';
  const allow = wrapAllow(rule, null);
  const key = (from, to) => from + '>' + to;
  const drawn = new Set([...coming, ...then].map(step => key(step.from, step.to)));
  const arrow = (step, kind) => {
    const at = squareXY(step.from, cell, view);
    const d = stepXY(wrapStep(step.from, step.to, W, H, allow), cell, view);
    const apart = drawn.has(key(step.to, step.from)) ? COMING_APART : 0;
    return comingArrow(at[0] + cell / 2, at[1] + cell / 2, d[0], d[1], cell, kind, apart);
  };
  let out = '';
  for (const step of then) out += arrow(step, 'comingThen');
  for (const step of coming) out += arrow(step, 'coming');
  return `<g data-coming="${coming.length}" data-coming-then="${then.length}"
      pointer-events="none">${out}</g>`;
}

// One step of a wild piece's walk (Q852), `(x1, y1)` to its tip at `(x2, y2)`: a shaft and
// a chevron in the walk's `colour`, laid on a casing of `--walk-edge` twice as wide, which
// is what holds it 3 : 1 on both woods whatever the hue. `dashed` for a jump between two
// squares that do not touch. `attrs` goes on the group (the data a reader finds it by).
const WALK_WIDTH = 0.075;
const WALK_HEAD = 0.17;
function walkLine(x1, y1, x2, y2, cell, colour, dashed, attrs) {
  const length = Math.hypot(x2 - x1, y2 - y1);
  if (!length) return '';
  const f = n => n.toFixed(3);
  const ux = (x2 - x1) / length, uy = (y2 - y1) / length;
  const head = Math.min(cell * WALK_HEAD, length * 0.6), width = cell * WALK_WIDTH;
  const back = [x2 - ux * head, y2 - uy * head];
  const left = [back[0] - uy * head * 0.75, back[1] + ux * head * 0.75];
  const right = [back[0] + uy * head * 0.75, back[1] - ux * head * 0.75];
  const shaft = `<line x1="${f(x1)}" y1="${f(y1)}" x2="${f(x2 - ux * width)}" y2="${f(y2 - uy * width)}"${
    dashed ? ` stroke-dasharray="${f(cell * 0.16)} ${f(cell * 0.13)}"` : ''}/>`;
  const chevron = `<path d="M${f(left[0])} ${f(left[1])}L${f(x2)} ${f(y2)}L${f(right[0])} ${f(right[1])}"/>`;
  return `<g ${attrs || ''} fill="none" stroke-linecap="round" stroke-linejoin="round"
      pointer-events="none"><g stroke="var(--walk-edge)" stroke-width="${f(width * 2.2)}">${shaft}${
    chevron}</g><g stroke="${colour}" stroke-width="${f(width)}">${shaft}${chevron}</g></g>`;
}

// The corner of a ring at that inset, where its beads sit. The rounded corner's arc at
// forty-five degrees, so a bead reads as strung on the ring it belongs to; `bottom`
// reflects it down the square, which is what keeps two kinds' beads apart on a square
// carrying both while their rings are kept apart by the inset.
const MARK_CORNER = 0.14;
function markCorner(inset, corner) {
  const side = 1 - 2 * inset, lean = MARK_CORNER * 0.707;
  const x = inset + side - MARK_CORNER + lean, y = inset + MARK_CORNER - lean;
  return corner === 'bottom' ? [x, 1 - y] : [x, y];
}

// The hex ring's bead point, in fractions of the cell's box: the right-hand vertex above
// the middle for `top`, below it for `bottom`.
function hexBead(inset, corner) {
  const scale = 1 - 2 * inset, r = scale / Math.sqrt(3), h = scale / 2;
  return [0.5 + h, corner === 'bottom' ? 0.5 + r / 2 : 0.5 - r / 2];
}

// Every mark a board puts on a square, keyed by square, each square carrying a **list**.
// More than one rule can single out the same square -- a goal square inside a promotion
// zone -- so a kind joins the list rather than replacing what is already there, and a
// square with nothing on it is absent rather than holding an empty list.
//
// Both renderers call this with their own ruleset block, which is why it reads named
// fields off one object: a new kind of mark is a line here and a row in `MARK_KINDS`.
//
// `hidden` names the kinds the reader has turned off -- `{ zone: true }` is the
// promotion zone's toggle -- and a kind that is off is left out whole rather than drawn
// faintly, because a square already carries the last move, a selection and a piece.
//
// `checked` is not a rule but a fact of the position on screen: the squares of the royal
// piece the engine reported in check (`checkSquares` on the play board). A still preview
// and the replay carry none.
function squareMarks(source, hidden) {
  const off = hidden || {};
  const marks = {};
  if (!off.check) addMarks(marks, 'check', checkMarks(source && source.checked));
  if (!off.goal) addMarks(marks, 'goal', goalMarks(source && source.goals));
  if (!off.zone) addMarks(marks, 'zone', zoneMarks(source && source.zones));
  return marks;
}

// Fold one kind's squares into the map, `found` being square -> the players it belongs
// to. Order is drawing order, so the kind added first is the outermost ring.
function addMarks(marks, kind, found) {
  for (const square of Object.keys(found)) {
    (marks[square] || (marks[square] = [])).push({ kind: kind, sides: found[square] });
  }
}

// Square -> the players a rule singles it out for, from that rule's compiled pair of
// per-player square lists. Every kind of mark is built this way, because "both sides,
// the same squares" and "each side its own end" are the two readings any per-player set
// has, and `null` -- a game without the rule at all -- is the third.
//
// The list of players is empty where naming one would say nothing: two sides on the
// same squares is one thing both are doing, and a bead per side there would only repeat
// the ring.
function sidedMarks(pairs) {
  const found = {};
  if (!pairs) return found;
  const first = pairs[0] || [], second = pairs[1] || [];
  const shared = first.length === second.length
    && first.every((square, i) => square === second[i]);
  const add = (square, player) => {
    const sides = found[square] || (found[square] = []);
    if (!shared) sides.push(player);
  };
  first.forEach(square => add(square, 0));
  second.forEach(square => add(square, 1));
  return found;
}

// The squares a game ends on when a piece reaches one, as `goal_payload` sends them.
// Racing kings and king of the hill send both sides to the same squares and name
// neither; dobutsu and the rps race mirror theirs, so each end of the board belongs to
// one player and the mark says which.
function goalMarks(goals) { return sidedMarks(goals); }

// The squares a piece promotes on, as `zone_payload` sends them. Chess mirrors its band
// like dobutsu mirrors its goal, so each end names its side; a game promoting both
// sides in one place names neither.
function zoneMarks(zones) { return sidedMarks(zones); }

// The squares a check is drawn on: a plain list, belonging to nobody in particular (the
// side is the side to move, which the status line names).
function checkMarks(squares) {
  const found = {};
  for (const square of squares || []) found[square] = [];
  return found;
}

// The pattern a hatched kind is filled with, once per drawing. A pattern is an element
// with an id, and an id is the document's, not the drawing's: the board, the legend's
// keys, a still preview and the replay each carry their own copy under their own `tag`,
// so none of them depends on another being on screen. Only the kinds the drawing
// actually carries are defined, so a board with no zone paints nothing in the zone's
// colour anywhere.
function markDefs(marks, tag) {
  const kinds = new Set();
  for (const square of Object.keys(marks || {})) {
    for (const mark of marks[square]) {
      if ((MARK_KINDS[mark.kind] || {}).shape === 'hatch') kinds.add(mark.kind);
    }
  }
  if (!kinds.size) return '';
  let out = '<defs>';
  for (const kind of kinds) {
    // In the drawing's own units, which are about 100 across every board, so the stripes
    // come out four or five to a square on any of them and in the legend's key.
    out += `<pattern id="${hatchId(kind, tag)}" patternUnits="userSpaceOnUse" width="2.6"
      height="2.6" patternTransform="rotate(45)"><rect width="0.8" height="2.6"
      fill="${MARK_KINDS[kind].colour}" opacity="0.38"/></pattern>`;
  }
  return out + '</defs>';
}

function hatchId(kind, tag) { return `mark-hatch-${kind}-${tag || 'board'}`; }

// The marks on one square: a ring per kind, with a bead on its corner per side the mark
// belongs to.
//
// A ring rather than a fill, because a fill is how the board already says *last move*
// and *ground held* and because a marker has to survive a piece standing on it -- which
// is the position that matters, since these squares are singled out for being arrived
// on. It sits inside the wild-piece trail's outline and in a colour of its own, so a
// board carrying both draws two rings that read as two things.
//
// The bead is filled in the player's own colour, which is how a board with no icons
// draws that player's pieces and how the hand and the seats name them. Drawn on the
// ring's corner rather than in the middle of the square, where a piece's head is -- and
// on the kind's own corner, so a goal square inside a promotion zone hangs one pair of
// beads at the top of the square and the other at the bottom.
//
// Its edge is the mark's colour and not `--ink`, because `--ink` changes ends between
// the themes: a dark bead edged in it reads as a *pale* bead under a dark theme, and the
// two sides then differ by a fill nobody looks at. In the mark's own colour the pair
// stays one filled bead against one all but empty, whichever way round the theme is.
//
// On a hex lattice (`view.hex`) the ring is the cell's hexagon at the kind's inset and the
// beads hang on its right-hand vertex, upper or lower by the kind's corner.
//
// A hatched kind fills the whole cell with its pattern (`markDefs`, under the same `tag`)
// and hangs no beads; a glow tints the whole cell. Both carry `data-mark`, so what a
// drawing shows can be read off it by kind.
function squareBadge(x, y, cell, marks, view, tag) {
  const hex = !!(view && view.hex);
  let out = '';
  for (const mark of marks) {
    const kind = MARK_KINDS[mark.kind] || MARK_KINDS.goal;
    if (kind.shape === 'hatch') {
      out += cellShape(x, y, cell, view, `fill="url(#${hatchId(mark.kind, tag)})"
           pointer-events="none" data-mark="${mark.kind}"`);
      continue;
    }
    if (kind.shape === 'edge') {
      out += `<line x1="${x + cell * 0.88}" y1="${y}" x2="${x + cell * 0.88}" y2="${y + cell}"
           stroke="${kind.colour}" stroke-width="${cell * 0.16}" pointer-events="none"
           data-mark="${mark.kind}"/>`;
      continue;
    }
    // Only the legend's key draws an arrow through here: on a board it runs between two
    // squares (`comingHtml`). Corner to corner, so the key reads as a step.
    if (kind.shape === 'arrow') {
      out += comingLine(x + cell * 0.2, y + cell * 0.8, x + cell * 0.8, y + cell * 0.2, cell,
                        mark.kind);
      continue;
    }
    // A standoff's key (schema 26): only the legend comes here; on a board the link runs
    // between the two squares (`standoffLine`). Corner to corner, as an arrow's key.
    if (kind.shape === 'link') {
      out += standoffLine(x + cell * 0.12, y + cell * 0.88, x + cell * 0.88, y + cell * 0.12,
                          cell, mark.kind);
      continue;
    }
    // A swap's meeting and a rested piece (schema 27): only the legend comes here; on a
    // board the burst lies between two squares and the sign on the rested piece's square.
    if (kind.shape === 'clash') {
      out += clashMark(x + cell * 0.1, y + cell * 0.9, x + cell * 0.9, y + cell * 0.1, cell);
      continue;
    }
    // An announced move (schema 27): only the legend comes here; on a board the arrow runs
    // between the two squares (`announcedHtml`). Corner to corner, as an arrow's key.
    if (kind.shape === 'announce') {
      out += announceLine(x + cell * 0.15, y + cell * 0.85, x + cell * 0.88, y + cell * 0.12, cell);
      continue;
    }
    if (kind.shape === 'rest') {
      out += restBadge(x - cell * 0.3, y + cell * 0.3, cell * 1.0);
      continue;
    }
    // A piece under an effect (schema 30): only the legend comes here, the badge drawn
    // larger and centred so the key reads; on a board it sits on its piece's corner.
    if (kind.shape === 'frozen' || kind.shape === 'disarmed') {
      const draw = kind.shape === 'frozen' ? frozenBadge : disarmedBadge;
      out += draw(x + cell * 0.1, y + cell * 0.1, cell * 2.0);
      continue;
    }
    if (kind.shape === 'shielded') {
      out += shieldedBadge(x + cell * 0.1, y - cell * 1.1, cell * 2.0);
      continue;
    }
    if (kind.shape === 'seen') {
      out += seenSquare(x, y, cell, SIGHT_FREEZE, view, false);
      continue;
    }
    if (kind.shape === 'order') {
      out += orderBadge(x + cell * 0.3, y - cell * 0.3, cell * 1.0, 1);
      continue;
    }
    // A walk's key, as a still board draws a step of one (Q852): only the legend comes here.
    if (kind.shape === 'walk') {
      out += walkLine(x + cell * 0.18, y + cell * 0.82, x + cell * 0.82, y + cell * 0.18, cell,
                      kind.colour, false, `data-mark="${mark.kind}"`);
      continue;
    }
    // A wall and a portal are drawn by the board's cell loop, never as a mark: only the
    // legend's key comes through here, the wall's square and a portal of the first pair.
    if (kind.shape === 'wall') {
      out += cellShape(x, y, cell, view, `fill="${kind.colour}" data-mark="${mark.kind}"`)
        + `<path d="${wallCross(x, y, cell, view)}" stroke="var(--ink)"
           stroke-width="${cell * 0.06}" opacity="0.5" pointer-events="none"/>`;
      continue;
    }
    // The portal's key is the game's own first end where the legend knows it (`mark.end`,
    // from `markKey`), else a fixed end of the first pair open to both.
    if (kind.shape === 'portal') {
      out += portalGlyph(mark.end || { pair: 1, end: 'orange', movedBy: [], open: [true, true] },
                         x, y, cell, { attrs: `data-mark="${mark.kind}"` });
      continue;
    }
    // The fold's key: the game's own first fold where the legend knows it, else a fixed
    // fold of the first group open to both.
    if (kind.shape === 'fold') {
      out += foldGlyph(mark.end || { pair: 1, end: 'orange', kind: 'fold', dest: 0, movedBy: [], open: [true, true] },
                       x, y, cell, { attrs: `data-mark="${mark.kind}"`, plain: true });
      continue;
    }
    if (kind.shape === 'glow') {
      out += cellShape(x, y, cell, view, `fill="${kind.colour}" opacity="0.55"
           pointer-events="none" data-mark="${mark.kind}"`);
      continue;
    }
    const inset = kind.inset, side = 1 - 2 * inset;
    const stroke = `fill="none" stroke="${kind.colour}" stroke-width="${cell * 0.055}"
           opacity="0.9" pointer-events="none" data-mark="${mark.kind}"`;
    out += hex
      ? `<path d="${hexPath(x, y, cell, inset)}" ${stroke}/>`
      : `<rect x="${x + cell * inset}" y="${y + cell * inset}"
           width="${cell * side}" height="${cell * side}" rx="${cell * MARK_CORNER}"
           ${stroke}/>`;
    const corner = hex ? hexBead(inset, kind.corner) : markCorner(inset, kind.corner);
    (mark.sides || []).forEach((player, index) => {
      const cx = x + cell * (corner[0] - index * 0.235), cy = y + cell * corner[1];
      out += `<circle cx="${cx}" cy="${cy}" r="${cell * 0.105}"
           fill="${player === 0 ? 'var(--p0)' : 'var(--p1)'}" stroke="${kind.colour}"
           stroke-width="${cell * 0.03}" pointer-events="none"/>`;
    });
  }
  return out;
}

// A kind of mark drawn small, for the legend line: the kind's own drawing on one square
// of the board's light shade, so the key *is* the mark rather than a description of it,
// and a kind restyled in `MARK_KINDS` restyles its key with it. `end` is a portal's key: the
// game's own first end (`portal_payload`), drawn as the board draws it.
function markKey(kind, end) {
  const marks = { 0: [{ kind: kind, sides: [], ...(end ? { end } : {}) }] };
  return `<svg class="key" viewBox="0 0 10 10" width="14" height="14" aria-hidden="true">${
    markDefs(marks, 'key')}<rect width="10" height="10" fill="var(--light)"/>${
    squareBadge(0, 0, 10, marks[0], null, 'key')}</svg>`;
}

// ----------------------------------------------------- what the reader turned off

// The promotion zone is on a toggle, and a toggle no browser remembers is one the
// reader turns off again on every board. `localStorage` is the right scope for it: how
// a board is read belongs to the person reading it, not to the game, so it never
// reaches the server and never travels with a ruleset. Both surfaces share the key, so
// a reader who put the zone away in the play board finds it away in a replay too.
//
// Every access is wrapped, because a browser set to refuse site data throws on the
// property itself rather than returning nothing, and a board that failed to draw over a
// preference is worse than a board that forgets one.
const ZONE_KEY = 'chess_variant_sandbox.zones';

// Shown by default. A promotion zone is a rule of the game that nothing else on the
// board shows -- the far rank looks like any other until a pawn reaches it -- so a
// reader who has never found the toggle should see the zone. The toggle is for putting
// it away once it has been read, which is the direction the memory then runs in.
function zonesShown() {
  try { return window.localStorage.getItem(ZONE_KEY) !== '0'; } catch (err) { return true; }
}

function rememberZones(shown) {
  try { window.localStorage.setItem(ZONE_KEY, shown ? '1' : '0'); } catch (err) { /* not offered */ }
}

// The control itself, so the two surfaces offer one toggle rather than two that drift.
// Tinted in the mark's own colour while it is on, which is what ties the button under
// the board to the rings on it.
function zoneToggleHtml(shown) {
  const lit = shown ? ' class="u-bc-zone u-c-zone"' : '';
  return `<button id="zonetoggle" aria-pressed="${!!shown}"${lit}
    title="the squares a piece promotes on">promotion zone</button>`;
}

// ------------------------------------------------------- the path a move takes

// One axis of a move, as the mechanic implies it. The direct reading wins unless the
// mover may leave by an edge and doing so is strictly shorter, so a tie is never drawn
// as a crossing and a board with edges always gets `to - from`.
//
// Presentation, not legality. A wrong guess makes a piece take a silly path across the
// board; it cannot make an illegal move look legal.
function axisStep(from, to, size, edges) {
  const direct = to - from;
  let step = direct, wrapped = false;
  if (edges && edges[1] && Math.abs(direct + size) < Math.abs(step)) {
    step = direct + size; wrapped = true;
  }
  if (edges && edges[0] && Math.abs(direct - size) < Math.abs(step)) {
    step = direct - size; wrapped = true;
  }
  return { step: step, wrapped: wrapped };
}

// The step a move takes, in signed squares across and up. The core supplies the two
// squares and, per player, which edges that player may leave by; this is the shortest
// step consistent with both. The two axes are decided separately, which is what lets a
// knight leap out of one corner of a torus and in at the next.
function wrapStep(from, to, W, H, allow) {
  // A hexagon's seams are sides, not axes: the step goes through a copy or not at all.
  if (allow && allow.copies) return hexagonStep(from, to, W, allow.copies);
  const files = axisStep(from % W, to % W, W, allow && allow.files);
  const ranks = axisStep(Math.floor(from / W), Math.floor(to / W), H, allow && allow.ranks);
  return { files: files.step, ranks: ranks.step,
           wrapped: files.wrapped || ranks.wrapped };
}

// A step as a translation on screen, which is where the flip lives: a board turned
// round runs its files the other way and its ranks down the page instead of up.
function stepXY(step, cell, view) {
  // On a hex lattice a rank also moves half a cell across, so the translation is read
  // off `cellXY` itself, which is affine: where the step lands less where it starts.
  if (view.hex) {
    const to = cellXY(step.files, step.ranks, cell, view), at = cellXY(0, 0, cell, view);
    return [to[0] - at[0], to[1] - at[1]];
  }
  return view.flipped
    ? [-step.files * cell, step.ranks * cell]
    : [step.files * cell, -step.ranks * cell];
}

// Whether the piece went over what stood between or through it. A straight unobstructed
// line travels, and everything else -- a knight, a cannon firing over a screen, a lame
// leap -- arcs. The squares in between are read through the seam where the step crosses
// one, so a rook sliding round the cylinder is still a slide.
//
// On a hexagon (`view.cut`) a square past a side is read off the copy there, which is
// the square the step really passed over.
function pathHops(board, from, step, W, H, view) {
  const dx = step.files, dy = step.ranks;
  if (dx !== 0 && dy !== 0 && Math.abs(dx) !== Math.abs(dy)) return true;
  const steps = Math.max(Math.abs(dx), Math.abs(dy));
  const sx = Math.sign(dx), sy = Math.sign(dy);
  const file = from % W, rank = Math.floor(from / W);
  const cut = view && view.cut;
  for (let i = 1; i < steps; i++) {
    const u = file + sx * i, v = rank + sy * i;
    if (cut) {
      const found = cellOnBoard(u, v, view) ? { square: v * W + u }
        : copyAt(u, v, view, cut.copies);
      if (found && board[found.square]) return true;
      continue;
    }
    if (board[wrapMod(v, H) * W + wrapMod(u, W)]) return true;
  }
  return false;
}

// A crossing move is drawn as the two copies the mechanic implies -- the piece walking
// off into the ghost extension, and its continuation walking in from the far side --
// which are one translation started a period apart. Both are ordinary travels, so
// neither renderer needs a second kind of motion; only the handover is stated here.
//
// The copy leaving fades out across the middle of the flight while the copy arriving
// fades in, so there is one piece to follow and never two. Without the handover a king
// appears to be in two places at once, which is the one thing a board must not say.
const CROSS_HANDOVER = [0.32, 0.68];
function crossOpacity(leaving) {
  const a = leaving ? 1 : 0, b = leaving ? 0 : 1;
  return [{ opacity: a }, { opacity: a, offset: CROSS_HANDOVER[0] },
          { opacity: b, offset: CROSS_HANDOVER[1] }, { opacity: b }];
}

// ---------------------------------------------------------------------- a portal
//
// Schema 19, drawn as Q838b built it and Q838f corrected it after the cold read
// (`docs/archive/plan-portal-look.md` §2.3, direction B, the vortex, and §6; the misreads are
// `docs/reference/portal-look-cold-read.md` §4): a portal end is drawn by the board, not taken from the icon library, because what
// it must say depends on the ruleset. One glyph, a disc inscribed in the cell (so it fits a
// hexagon's cell when portals reach one), every part of it the portal itself:
//
//   well      a hole in the pair's colour (`--pair-N`, the same at both ends: colour means
//             the pair), falling to the near-black keyline's colour (`--portal-edge`) in the
//             middle, and ringed in that keyline so it holds on both woods;
//   arms      who may go in here (the payload's `open`): two spiral arms turning down into
//             the eye, in the colours of the pieces that may go in -- one white and one black
//             for both players, two of one colour for one. Nobody: no arms at all, a still
//             well things only come out of (Q838f: faint dark arms were taken for Black's). The
//             arms stop short of the rim (`PORTAL_REACH`), so the well's outermost ring is
//             always the pair's colour and never an arm's: a white arm reaching the rim of a
//             sunk well read as a white token's collar. A white arm is edged dark and a black
//             one in the pair's colour, never light, so a black arm's edge is never taken for
//             a white arm;
//   stand     who may move it (`movedBy`): a token in the mover's colour under the well,
//             raised like a piece with its shadow below it, a wide collar round a smaller well
//             (half and half for both, the bottom half the player at the bottom of the board
//             as drawn); for nobody, no token, the well sunk into the wood: a hole, the shadow
//             of its top edge falling inside it (`pg-hole`), and no light rim anywhere (Q838f:
//             the old lip's light lower edge read as a collar);
//   number    the pair's, in the dark eye, in the pair's colour, where the game has more
//             than one pair;
//   beak      where it leads (schema 21, `docs/archive/plan-portals-2.md` §5.1, Q843b): a pointed
//             notch on the rim in the group's colour, cased in the keyline, aimed at the end
//             it leads to as drawn (`options.aim`, `portalAim`). Only on an end somebody may
//             go in by: a way out leads nowhere a piece goes from. A pair's two beaks point at
//             each other, a ring's chase each other round, a funnel's all point at its way
//             out. At or above `PORTAL_FINE_FROM` px a cell it stands outside the well
//             (`pg-beak`), from `PORTAL_DETAIL_FROM` it is a wedge cut into the rim
//             (`pg-beak-cut`), and under that neither is drawn: the group map carries it;
//   filter    an end that lets only some pieces in (the payload's `filtered`, schema 21's
//             `carries`): its arms are dashed, each broken into three strokes (§5.2). With
//             `options.dim` (a picked piece it refuses) its arms and beak are dimmed to
//             `PORTAL_DIM`, so the doors that piece may use are the bright ones;
//   idle      on a live board (`options.idle`), the arms turn inwards once every
//             `PORTAL_TURN` s, the two ends of a pair in step and pairs out of step
//             (`portalPhase`). The turn is the stylesheet's (`.pg-idle`), on the arms' group
//             inside the glyph, so the trip's pulse on the outer group (Q836) composes with
//             it, and it is off under reduced motion. An end nobody goes in by has no arms,
//             so nothing on it turns.
//
// Every fact is the compiled payload's (`viz.wrapped.portal_payload`); nothing here reads
// the DSL or works out where a piece goes. The parts carry classes so a small board can
// change them (D9, the board's `data-portals` tier): under `PORTAL_FINE_FROM` px a cell the
// arms' edges are held at one screen pixel, which is what keeps a black arm on violet;
// under `PORTAL_DETAIL_FROM` the arms (`pg-arm`) give way to two broad wedges
// (`pg-wedge`); the number (`pg-num`) is never drawn smaller than `PORTAL_NUMBER_LEAST` px,
// and its eye (`pg-eye`) grows to hold it. The token and the hole are drawn at every size.
const PORTAL_FINE_FROM = 28;
const PORTAL_DETAIL_FROM = 20;
const PORTAL_NUMBER_LEAST = 10;
// The glyph's radii, in cells: the token, the well on a token and sunk, the eye, and the
// keyline round the well. The token's collar is the difference, 0.17 of a cell.
const PORTAL_TOKEN = 0.46;
const PORTAL_WELL_RAISED = 0.29;
const PORTAL_WELL_SUNK = 0.38;
const PORTAL_EYE = 0.13;
const PORTAL_KEYLINE = 0.024;
// How far out the arms reach, as a share of the well's radius (the rest is the pair's ring);
// their width at the widest, as a share of it; and how deep the hole's shadow falls inside a
// sunk well, in cells.
const PORTAL_REACH = 0.78;
const PORTAL_ARM_WIDTH = 0.42;
const PORTAL_HOLE = 0.1;
// The pair's number, as a share of a cell; and the eye's least radius, as a share of the
// number's least size, wide enough to hold a digit when the number is held up.
const PORTAL_NUMBER = 0.24;
const PORTAL_EYE_LEAST = 0.55;
// One turn of the arms, in seconds (the stylesheet's `.pg-idle` says the same).
const PORTAL_TURN = 14;
// The arms drawn by mass (Q838h, `docs/reference/portal-look-cold-read.md` §7.5): at the mid
// tier and on a phone (the stylesheet's `pg-mass`), each arm this share of the well's
// radius at its widest, a white arm edged in the group's colour rather than dark, so it is
// never a white sliver in a dark outline that reads as a dark arm.
const PORTAL_MASS_WIDTH = 0.62;
// The beak, in cells: outside the well at the full size, its tip `PORTAL_BEAK_OUT` beyond
// the well's rim and its base `PORTAL_BEAK_HALF` either side of its line; at the mid size a
// wedge cut into the rim, from `PORTAL_CUT_IN` inside it to `PORTAL_CUT_OUT` outside.
const PORTAL_BEAK_OUT = 0.13;
const PORTAL_BEAK_HALF = 0.1;
const PORTAL_CUT_IN = 0.11;
const PORTAL_CUT_OUT = 0.07;
const PORTAL_CUT_HALF = 0.1;
// How far an end a picked piece may not go in by is dimmed: its arms and its beak.
const PORTAL_DIM = 0.35;
// A filtered end's arm in three strokes: the stretches of its length drawn, from the rim (0)
// to the eye (1).
const PORTAL_DASHES = [[0, 0.27], [0.365, 0.635], [0.73, 1]];
// The pieces' colours.
const PORTAL_SIDE = ['var(--p0)', 'var(--p1)'];
let PORTAL_WELLS = 0;

// The end this one leads to (schema 21's `dest`), read off the same payload: for a pair,
// its partner; for an end of a map, the end its `to` names; null for a way out only. A
// payload from before schema 21 has no `dest`, and the partner is the pair's other end.
function portalPartner(portals, end) {
  if (!portals || !end) return null;
  if (end.dest !== undefined) {
    return end.dest === null ? null : portals.ends.find(other => other.type === end.dest) || null;
  }
  return portals.ends.find(other => other.pair === end.pair && other.end !== end.end) || null;
}

// The ends that lead to this one (schema 21): the ways in to the square a piece comes out
// beside. Empty for a payload from before schema 21.
function portalSources(portals, end) {
  if (!portals || !end) return [];
  return portals.ends.filter(other => other.dest !== undefined && other.dest === end.type);
}

// Whether an end belongs to a map rather than a pair (schema 21): it leads nowhere, or the
// end it leads to does not lead back. A pair's words are the ones it has always had.
function portalMapped(end, partner) {
  return !!end && (end.dest === null
    || (!!partner && partner.dest !== undefined && partner.dest !== end.type));
}

// Squares or piece ids as a list in a sentence: *b3*, *b3 and c6*, *b3, c6 and f2*.
function portalList(names) {
  return names.length < 2 ? (names[0] || '')
    : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

// The filter's words (schema 21's `carries`): which pieces may go in here, where the end
// names them (*Only knights and bishops may go in here*, §5.2); empty otherwise. `nameOf`
// names a piece id as the ruleset calls its pieces, many of them (`portalNamer`); without
// it the ids are said.
function portalFilterWords(end, nameOf) {
  if (!end || !end.filtered) return '';
  const ids = [...new Set([...((end.admits || [])[0] || []), ...((end.admits || [])[1] || [])])];
  const said = ids.map(id => (nameOf && nameOf(id)) || id);
  return said.length ? ` Only ${portalList(said)} may go in here.` : '';
}

// A piece's name, many of it: *knights*, *bishops*, *ferzes*.
function portalPlural(name) {
  const word = String(name || '');
  if (!word) return word;
  if (/(s|x|z|ch|sh)$/i.test(word)) return word + 'es';
  if (/[^aeiou]y$/i.test(word)) return word.slice(0, -1) + 'ies';
  return word + 's';
}

// `nameOf` for `portalFilterWords`, from a payload's piece ids and names in type order.
function portalNamer(pieceIds, pieceNames) {
  if (!pieceIds || !pieceNames) return null;
  return id => {
    const at = pieceIds.indexOf(id);
    return at >= 0 && pieceNames[at] ? portalPlural(pieceNames[at]) : null;
  };
}

// Whether the page is on a phone's width, where the arms are drawn by mass (Q838h).
const PORTAL_PHONE = 480;
function portalPhone() {
  try {
    return typeof window !== 'undefined' && window.innerWidth > 0 && window.innerWidth <= PORTAL_PHONE;
  } catch (err) { return false; }
}

// How many pairs the game has, which is whether an end shows its number.
function portalPairs(portals) {
  return portals ? new Set(portals.ends.map(end => end.pair)).size : 0;
}

// How one player's pieces may use this end: `both` (in here or in at the partner),
// `in` (in here only), `out` (they only come out beside it) or `closed`, as the words
// say it (`portalWords`). Null for a player who is neither seat.
function portalPassage(end, partner, player) {
  if (player !== 0 && player !== 1) return null;
  const here = !!(end.open || [])[player];
  const there = !!(partner && (partner.open || [])[player]);
  return here && there ? 'both' : here ? 'in' : there ? 'out' : 'closed';
}

// What the glyph keeps at a cell this many pixels across: `full`, `mid` (no lip, the arms'
// edges a pixel) or `small` (the arms as two wedges). Unknown, it is drawn whole.
function portalTier(px) {
  if (!(px > 0) || px >= PORTAL_FINE_FROM) return 'full';
  return px >= PORTAL_DETAIL_FROM ? 'mid' : 'small';
}

// Where a pair's arms are in their turn, as the turn's `animation-delay`, in seconds: read
// off the document's clock, so a board drawn again carries on from where it was rather than
// jumping back, and offset by the pair's number, so the two ends of a pair turn in step and
// different pairs out of step with each other.
function portalPhase(pair) {
  let now = 0;
  try {
    const timeline = typeof document !== 'undefined' && document.timeline;
    now = Number(timeline && timeline.currentTime) || 0;
  } catch (err) { now = 0; }
  const at = (now / 1000 + pair * 5.3) % PORTAL_TURN;
  return -at;
}

// One spiral arm as a filled, tapering shape: from radius `outer` at angle `start` in to
// radius `inner`, turning `sweep` radians on the way (clockwise on screen), `width` at its
// widest and tapering at both ends. `from` and `to` (0 to 1, the whole arm by default) draw
// only that stretch of it, its taper the whole arm's, so three stretches are a dashed arm.
function portalArm(cx, cy, outer, inner, start, sweep, width, from, to) {
  const t0 = from === undefined ? 0 : from, t1 = to === undefined ? 1 : to;
  const steps = Math.max(4, Math.round(18 * (t1 - t0))), left = [], right = [];
  for (let i = 0; i <= steps; i++) {
    const t = t0 + (t1 - t0) * i / steps;
    const r = outer + (inner - outer) * t, a = start + sweep * t;
    const r2 = outer + (inner - outer) * (t + 0.01), a2 = start + sweep * (t + 0.01);
    const x = cx + r * Math.cos(a), y = cy + r * Math.sin(a);
    const dx = cx + r2 * Math.cos(a2) - x, dy = cy + r2 * Math.sin(a2) - y;
    const len = Math.hypot(dx, dy) || 1;
    const w = width * (0.25 + 0.75 * Math.sin(Math.PI * Math.min(1, t * 1.15 + 0.08)));
    left.push([x - dy / len * w / 2, y + dx / len * w / 2]);
    right.push([x + dy / len * w / 2, y - dx / len * w / 2]);
  }
  const points = [...left, ...right.reverse()];
  return 'M' + points.map(p => `${p[0].toFixed(3)} ${p[1].toFixed(3)}`).join('L') + 'Z';
}

// An arm whole, or (`dashed`, a filtered end's) as its three strokes in one path.
function portalArmPath(cx, cy, outer, inner, start, sweep, width, dashed) {
  if (!dashed) return portalArm(cx, cy, outer, inner, start, sweep, width);
  return PORTAL_DASHES.map(([from, to]) =>
    portalArm(cx, cy, outer, inner, start, sweep, width, from, to)).join('');
}

// Where a beak points: the angle on screen, in radians (0 to the right, a quarter turn
// clockwise straight down, as SVG's y runs), from an end drawn centred on (ax, ay) to the
// end it leads to drawn centred on (bx, by); null where the two are drawn on one spot. Pure,
// of two drawn positions, so where a beak points is something a test can hold.
function portalAim(ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  if (!(Math.hypot(dx, dy) > 1e-9)) return null;
  return Math.atan2(dy, dx);
}

// The centre of the copy of `square` drawn nearest the point (cx, cy) -- the square itself
// or a ghost of it round a wrapping board (the beak and the map aim at the nearest copy,
// §5.1) -- in a drawing whose cells are `cell` across in `view`.
function portalNearestCentre(square, cx, cy, cell, view) {
  const centre = (u, v) => { const xy = cellXY(u, v, cell, view); return [xy[0] + cell / 2, xy[1] + cell / 2]; };
  const own = squareCell(square, view);
  let best = centre(own[0], own[1]);
  let far = Math.hypot(best[0] - cx, best[1] - cy);
  if (view && (view.files.rings || view.ranks.rings || view.cut)) {
    for (const ghost of ghostCells(view)) {
      if (ghost.square !== square) continue;
      const at = centre(ghost.u, ghost.v);
      const d = Math.hypot(at[0] - cx, at[1] - cy);
      if (d < far) { best = at; far = d; }
    }
  }
  return best;
}

// The beak's aim for an end drawn in the cell whose corner is (x, y), leading to the end
// standing on `there` (-1: not on the board), or null for none: an end nobody may go in by
// has no beak, since nothing goes from it.
function portalAimFor(end, there, x, y, cell, view) {
  if (!end || !(there >= 0) || !(end.open || []).some(Boolean)) return null;
  const cx = x + cell / 2, cy = y + cell / 2;
  const [bx, by] = portalNearestCentre(there, cx, cy, cell, view);
  return portalAim(cx, cy, bx, by);
}

// One arrow of the group map (the play board's and the designer's bench's), from the end drawn at (ax, ay) to the end drawn at (bx, by):
// bowed to the left of its way, so a pair's two arrows, one each way, never lie on each
// other; held back from both glyphs, its head at the end it leads to. `pair` is the group's
// number (its colour).
function portalArrowHtml(ax, ay, bx, by, cell, pair, said) {
  const length = Math.hypot(bx - ax, by - ay);
  if (!(length > cell)) return '';
  const ux = (bx - ax) / length, uy = (by - ay) / length;
  const bow = Math.min(length * 0.2, cell * 1.2);
  const qx = (ax + bx) / 2 + uy * bow, qy = (ay + by) / 2 - ux * bow;
  const toward = (px, py, r) => {
    const d = Math.hypot(qx - px, qy - py) || 1;
    return [px + (qx - px) / d * r, py + (qy - py) / d * r];
  };
  const [sx, sy] = toward(ax, ay, cell * 0.46);
  const [ex, ey] = toward(bx, by, cell * 0.5);
  // The head, along the curve's own direction where it ends.
  const hx = ex - qx, hy = ey - qy, hl = Math.hypot(hx, hy) || 1;
  const tx = hx / hl, ty = hy / hl, size = cell * 0.2;
  const n = v => v.toFixed(3);
  const head = `M${n(ex + tx * size * 0.35)} ${n(ey + ty * size * 0.35)}L${n(ex - tx * size + ty * size * 0.6)} ${
    n(ey - ty * size - tx * size * 0.6)}L${n(ex - tx * size - ty * size * 0.6)} ${n(ey - ty * size + tx * size * 0.6)}Z`;
  const d = `M${n(sx)} ${n(sy)}Q${n(qx)} ${n(qy)} ${n(ex)} ${n(ey)}`;
  return `<g data-way="${said}"><path d="${d}" fill="none" stroke="var(--portal-edge)" stroke-width="${
    n(cell * 0.12)}" stroke-linecap="round" opacity="0.6"/><path d="${d}" fill="none" stroke="var(--pair-${pair})"
    stroke-width="${n(cell * 0.055)}" stroke-linecap="round"/><path d="${head}" fill="var(--pair-${pair})"
    stroke="var(--portal-edge)" stroke-width="${n(cell * 0.03)}" stroke-linejoin="round" paint-order="stroke"/></g>`;
}

// One beak as a filled triangle along `angle` from the centre (cx, cy): its base `base`
// out, `half` either side, its tip `tip` out.
function portalBeakPath(cx, cy, angle, base, tip, half) {
  const ux = Math.cos(angle), uy = Math.sin(angle);
  const n = v => v.toFixed(3);
  const bx = cx + ux * base, by = cy + uy * base;
  return `M${n(cx + ux * tip)} ${n(cy + uy * tip)}L${n(bx - uy * half)} ${n(by + ux * half)}L${
    n(bx + uy * half)} ${n(by - ux * half)}Z`;
}

// The glyph for one end, as SVG, in the cell whose corner is (x, y). `options.numbered`
// draws the pair's number; `options.label` is its words for a screen reader;
// `options.attrs` goes on the group (the board's `data-piece`); `options.idle` turns its
// arms (a live board's, never a still's); `options.flipped` says the board is drawn turned
// round, so the second player is at the bottom; `options.aim` is where its beak points
// (`portalAim`; absent or null, no beak); `options.dim` dims its arms and beak (a picked
// piece it refuses, §5.2).
function portalGlyph(end, x, y, cell, options) {
  const opts = options || {};
  const cx = x + cell / 2, cy = y + cell / 2;
  const hue = `var(--pair-${end.pair})`;
  const edge = 'var(--portal-edge)';
  const movers = end.movedBy || [];
  const open = end.open || [];
  const low = opts.flipped ? 1 : 0;
  const n = v => v.toFixed(3);
  const circle = (r, attrs) => `<circle cx="${n(cx)}" cy="${n(cy)}" r="${n(cell * r)}" ${attrs}/>`;
  const dim = opts.dim ? ` opacity="${PORTAL_DIM}"` : '';
  const aim = typeof opts.aim === 'number' && isFinite(opts.aim) && open.some(Boolean) ? opts.aim : null;
  const beak = (cls, base, tip, half) => `<path class="${cls}" d="${portalBeakPath(cx, cy, aim,
    cell * base, cell * tip, cell * half)}" fill="${hue}" stroke="${edge}" stroke-width="${
    n(cell * 0.03)}" stroke-linejoin="round" paint-order="stroke"${dim}/>`;
  let out = '';
  const well = movers.length ? PORTAL_WELL_RAISED : PORTAL_WELL_SUNK;
  if (movers.length) {
    // The token, raised: its shadow falls below it, as a piece's would, and its colour is
    // whose it is to move, as a piece's is.
    const r = cell * PORTAL_TOKEN;
    out += `<g class="pg-token" data-moved="${movers.join(' ')}">`;
    out += `<circle cx="${n(cx + cell * 0.025)}" cy="${n(cy + cell * 0.06)}" r="${n(r)}"
      fill="${edge}" opacity="0.45"/>`;
    if (movers.includes(0) && movers.includes(1)) {
      // Half and half: the bottom half the player at the bottom of the board as drawn.
      out += `<path d="M${n(cx - r)} ${n(cy)}A${n(r)} ${n(r)} 0 0 0 ${n(cx + r)} ${n(cy)}Z"
        fill="${PORTAL_SIDE[low]}"/>`;
      out += `<path d="M${n(cx - r)} ${n(cy)}A${n(r)} ${n(r)} 0 0 1 ${n(cx + r)} ${n(cy)}Z"
        fill="${PORTAL_SIDE[1 - low]}"/>`;
    } else {
      out += circle(PORTAL_TOKEN, `fill="${PORTAL_SIDE[movers.includes(0) ? 0 : 1]}"`);
    }
    out += circle(PORTAL_TOKEN, `fill="none" stroke="${edge}" stroke-width="${n(cell * PORTAL_KEYLINE)}"`);
    out += '</g>';
  }
  // The beak at the full size, outside the well (over a token's collar), drawn before the
  // well so the well's rim covers its base.
  if (aim !== null) {
    out += beak('pg-beak', well - 0.04, well + PORTAL_BEAK_OUT, PORTAL_BEAK_HALF);
  }
  // The well: the pair's colour out to the rim from 45% of the radius, falling to the edge's
  // near-black in the eye, from 14% in (Q838f: the old well, dark out to 78%, hid a black arm
  // on violet but for its light edge, which then read as a white arm).
  const id = `pgwell${++PORTAL_WELLS}`;
  out += `<defs><radialGradient id="${id}"><stop offset="0" stop-color="${edge}"/>
    <stop offset="0.14" stop-color="${edge}"/><stop offset="0.45" stop-color="${hue}"/>
    <stop offset="1" stop-color="${hue}"/></radialGradient></defs>`;
  out += circle(well, `class="pg-well" fill="url(#${id})"`);
  // The arms: whose pieces may go in here. Both: the arm from the bottom is the bottom
  // player's. One: two of that colour. Nobody: none, and nothing turns.
  const who = [0, 1].filter(p => open[p]);
  const colours = who.length === 2 ? [low, 1 - low] : who.length === 1 ? [who[0], who[0]] : [];
  const edges = [edge, hue];
  const dashed = !!end.filtered;
  // On a half-and-half token each arm starts beside the collar's other half (Q838h, §7.5 of
  // the cold read): the bottom player's arm from the top, by the top player's half, so
  // neither arm lies beside its own colour and vanishes into it on a still.
  const split = movers.includes(0) && movers.includes(1) && colours.length === 2 ? 1 : 0;
  let arms = '';
  colours.forEach((p, i) => {
    const start = Math.PI * 0.5 + (i + split) * Math.PI;
    const paint = `fill="${PORTAL_SIDE[p]}" stroke="${edges[p]}" stroke-width="${
      n(cell * 0.018)}" stroke-linejoin="round"`;
    const said = ` data-arm="${p}"`;
    arms += `<path class="pg-arm"${said} d="${portalArmPath(cx, cy, cell * well * PORTAL_REACH,
      cell * 0.05, start, Math.PI * 1.25, cell * well * PORTAL_ARM_WIDTH, dashed)}" ${paint}/>`;
    // By mass (Q838h): the mid tier's and a phone's, wider, a white arm edged in the
    // group's colour.
    arms += `<path class="pg-mass"${said} d="${portalArmPath(cx, cy, cell * well * PORTAL_REACH,
      cell * 0.05, start, Math.PI * 1.25, cell * well * PORTAL_MASS_WIDTH, dashed)}" fill="${
      PORTAL_SIDE[p]}" stroke="${hue}" stroke-width="${n(cell * 0.018)}" stroke-linejoin="round"/>`;
    // The small tier's: a broad wedge, a quarter turn, which still reads as a swirl when
    // the arm would be a hairline.
    arms += `<path class="pg-wedge"${said} d="${portalArmPath(cx, cy, cell * well * 0.68,
      cell * 0.04, start, Math.PI * 0.75, cell * well * 0.75, dashed)}" ${paint}/>`;
  });
  if (colours.length) {
    const idle = opts.idle
      ? ` pg-idle ${styleClass(`transform-origin:${n(cx)}px ${n(cy)}px; animation-delay:${
        portalPhase(end.pair).toFixed(2)}s`)}`
      : '';
    out += `<g class="pg-arms${idle}" data-turn="in"${
      dashed ? ' data-filtered=""' : ''}${dim}>${arms}</g>`;
  }
  if (!movers.length) {
    // Sunk: a hole cut into the board, lit from above, so the shadow of its top edge falls
    // inside it, over the arms. Nothing light rings it: that is a token's collar.
    const r = cell * well, d = cell * PORTAL_HOLE;
    const sx = Math.sqrt(r * r - d * d / 4), sy = cy + d / 2;
    out += `<path class="pg-hole" d="M${n(cx - sx)} ${n(sy)}A${n(r)} ${n(r)} 0 1 1 ${n(cx + sx)} ${
      n(sy)}A${n(r)} ${n(r)} 0 0 0 ${n(cx - sx)} ${n(sy)}Z" fill="${edge}" opacity="0.7"/>`;
  }
  out += circle(well, `fill="none" stroke="${edge}" stroke-width="${n(cell * PORTAL_KEYLINE)}"`);
  // The beak at the mid size: a wedge cut into the rim, over it.
  if (aim !== null) {
    out += beak('pg-beak-cut', well - PORTAL_CUT_IN, well + PORTAL_CUT_OUT, PORTAL_CUT_HALF);
  }
  if (opts.numbered) {
    out += circle(PORTAL_EYE, `class="pg-eye" fill="${edge}" data-em="${PORTAL_EYE}"`);
    out += `<text class="pg-num" x="${n(cx)}" y="${n(cy)}" font-size="${n(cell * PORTAL_NUMBER)}"
      data-em="${PORTAL_NUMBER}" fill="${hue}" font-weight="700" text-anchor="middle"
      dominant-baseline="central">${end.pair}</text>`;
  }
  const label = opts.label ? ` role="img" aria-label="${String(opts.label).replace(/"/g, '&quot;')}"` : '';
  const said = aim !== null ? ` data-aim="${(aim * 180 / Math.PI).toFixed(1)}"` : '';
  return `<g data-portal="${end.id || end.pair + end.end[0]}" data-pair="${end.pair}"${said}${
    opts.dim ? ' data-dim=""' : ''}${label}${opts.attrs ? ' ' + opts.attrs : ''}>${out}</g>`;
}

// ---------------------------------------------------------------------- a fold
//
// Schema 22's fold portal (`docs/archive/plan-portals-2.md` §5.3, Q843d): the pool, the face-on
// version of a luminous portal, lying in the board. The vortex is a hole things fall into;
// the pool is a surface light comes through, so the two are told apart at a glance and in a
// grey picture (one is dark in the middle, the other bright). It is drawn in the board's
// square layer, after the square's tints and before its piece, because a piece stands in it:
//
//   pool      a disc 0.9 of a cell across in the group's colour, near white at the centre
//             (`--fold-light`) through the hue to a saturated, ragged rim; the rim's edge is
//             displaced by turbulence (`pf-rim`), which is what reads as liquid, not a disc;
//             and a soft glow in the hue that spills onto the neighbouring squares
//             (`foldGlowHtml`, laid over the board after its squares, since a later square
//             would cover it);
//   bands     who may use it (the payload's `open`): two soft spiral bands turning in it, one
//             white where White's pieces may, one near-black where Black's may (two of one for
//             one side): the vortex's arms in the pool's own medium, drawn by mass (§5.3 asked
//             for bands of the hue edged in the side's colour; two cold readers could not read
//             an edge at a board's size, `docs/reference/portal-look-cold-read.md` §8). A fold that leads nowhere has plain bands and turns outward. A
//             filtered end's bands are dashed (§5.2), and `options.dim` dims them and the beak;
//   collar    who may move it (`movedBy`): a ring just inside the cell's edge in the mover's
//             colour (half and half for both, the bottom half the player at the bottom of
//             the board as drawn). The pool cannot stand on a token, since a piece stands in
//             it. A fixed fold has none;
//   beak      where it leads, as on the vortex (`options.aim`, `portalAim`; `pg-beak` at the
//             full size, `pg-beak-cut` at the mid, neither smaller), on the rim;
//   number    the group's, in the bright centre in the keyline's near-black, where the game
//             has more than one group; with a piece standing in the fold (`options.occupied`)
//             it moves to the cell's top-right corner as a badge (`foldBadgeHtml`), drawn
//             over the piece so it is not hidden;
//   idle      on a live board (`options.idle`), the bands turn once every `FOLD_TURN` s and
//             the rim churns once every `FOLD_WOBBLE` s, a group's ends in step
//             (`foldPhase`); the stylesheet's (`.pf-idle`, `.pf-wobble`), off under reduced
//             motion.
//
// Where a fold leads is said by its beak and the portal map, as for a momentum end; since
// schema 24 nothing is drawn on the far end (`docs/archive/plan-portals-2.md` §5.3R: the echo went).
// Every fact is the compiled payload's (`portal_payload`, `fold_payload`); nothing here works
// out a move.
const FOLD_POOL = 0.45;
const FOLD_KEYLINE = 0.018;
const FOLD_RIM = 0.07;
const FOLD_COLLAR = 0.44;
const FOLD_COLLAR_WIDTH = 0.11;
// The glow, a ring in the hue from inside the rim out past the cell's edge, in cells.
const FOLD_GLOW_IN = 0.4;
const FOLD_GLOW_OUT = 0.64;
// The bands, as a share of the pool's radius: how far out they start, how far in they end,
// and their width at the widest.
const FOLD_BAND_OUT = 0.9;
const FOLD_BAND_IN = 0.1;
const FOLD_BAND_WIDTH = 0.5;
// A band's edge, in cells.
const FOLD_EDGE = 0.022;
// One turn of the bands and one churn of the rim, in seconds (the stylesheet's say the same).
const FOLD_TURN = 12;
const FOLD_WOBBLE = 8;
// The number's badge with a piece standing in the fold: its radius and centre, in cells.
const FOLD_BADGE = 0.15;
const FOLD_BADGE_AT = [0.84, 0.16];
let FOLD_POOLS = 0;

// Whether a payload end is a fold (schema 22's `kind`); every end before it is momentum.
function isFold(end) { return !!end && end.kind === 'fold'; }

// A group's place in its bands' turn, as `portalPhase` is for the arms: off the document's
// clock and the group's number, so every end of a group turns in step.
function foldPhase(pair) {
  let now = 0;
  try {
    const timeline = typeof document !== 'undefined' && document.timeline;
    now = Number(timeline && timeline.currentTime) || 0;
  } catch (err) { now = 0; }
  return -((now / 1000 + pair * 4.1) % FOLD_TURN);
}

function foldGlyph(end, x, y, cell, options) {
  const opts = options || {};
  const cx = x + cell / 2, cy = y + cell / 2;
  const hue = `var(--pair-${end.pair})`;
  const edge = 'var(--portal-edge)';
  const movers = end.movedBy || [];
  const open = end.open || [];
  const low = opts.flipped ? 1 : 0;
  const n = v => v.toFixed(3);
  const r = cell * FOLD_POOL;
  const circle = (radius, attrs) => `<circle cx="${n(cx)}" cy="${n(cy)}" r="${n(radius)}" ${attrs}/>`;
  const dim = opts.dim ? ` opacity="${PORTAL_DIM}"` : '';
  const aim = typeof opts.aim === 'number' && isFinite(opts.aim) && open.some(Boolean) ? opts.aim : null;
  const id = ++FOLD_POOLS;
  let out = `<defs><radialGradient id="pfcore${id}"><stop offset="0" stop-color="var(--fold-light)" stop-opacity="1"/>
    <stop offset="0.36" stop-color="var(--fold-light)" stop-opacity="0.95"/>
    <stop offset="0.8" stop-color="var(--fold-light)" stop-opacity="0"/></radialGradient>`;
  // The rim's liquid edge: turbulence displacing it, scaled to the cell (`baseFrequency` is
  // in the drawing's units). A thumbnail keeps the plain rim.
  if (!opts.plain) {
    out += `<filter id="pfwob${id}" x="-25%" y="-25%" width="150%" height="150%">
      <feTurbulence type="fractalNoise" baseFrequency="${(3 / cell).toFixed(4)}" numOctaves="2" seed="${end.pair * 7 + 3}"/>
      <feDisplacementMap in="SourceGraphic" scale="${n(cell * 0.11)}" xChannelSelector="R" yChannelSelector="G"/></filter>`;
  }
  out += '</defs>';
  // The pool: the hue, lit from within.
  out += circle(r, `class="pf-pool" fill="${hue}"`);
  out += circle(r, `fill="url(#pfcore${id})"`);
  // The bands: whose pieces may use it. Both: the bottom player's edge first.
  const who = [0, 1].filter(p => open[p]);
  const edges = who.length === 2 ? [low, 1 - low] : who.length === 1 ? [who[0], who[0]] : [null, null];
  const dashed = !!end.filtered;
  const turn = who.length ? 'in' : 'out';
  let bands = '';
  edges.forEach((p, i) => {
    const start = Math.PI * 0.5 + i * Math.PI;
    const d = portalArmPath(cx, cy, r * FOLD_BAND_OUT, r * FOLD_BAND_IN, start, Math.PI * 1.2,
                            r * FOLD_BAND_WIDTH, dashed);
    // By mass (the second cold read, `docs/reference/portal-look-cold-read.md` §8.5: an edge
    // did not read at a board's size): a band in the colour of the pieces that may use it, a
    // white one edged in the keyline so it holds against the pool's light middle, a near-black
    // one in the hue; a fold that leads nowhere has bands of its own hue.
    const paint = p === null ? `fill="${hue}" fill-opacity="0.6" stroke="none"`
      : `fill="${PORTAL_SIDE[p]}" fill-opacity="0.92" stroke="${p === 0 ? edge : hue}" stroke-width="${
        n(cell * FOLD_EDGE)}" stroke-linejoin="round"`;
    bands += `<path class="pf-band"${p === null ? '' : ` data-band="${p}"`} d="${d}" ${paint}/>`;
  });
  const spin = opts.idle
    ? ` pf-idle ${styleClass(`transform-origin:${n(cx)}px ${n(cy)}px; animation-delay:${
      foldPhase(end.pair).toFixed(2)}s`)}` : '';
  out += `<g class="pf-bands${spin}" data-turn="${turn}"${
    dashed ? ' data-filtered=""' : ''}${dim}>${bands}</g>`;
  // The rim: saturated, ragged and churning, cased in a faint keyline so it holds on both
  // woods (no hue holds against both).
  const wobble = opts.idle
    ? `pf-wobble ${styleClass(`transform-origin:${n(cx)}px ${n(cy)}px`)}` : '';
  const lumps = `stroke-dasharray="${n(cell * 0.16)} ${n(cell * 0.09)} ${n(cell * 0.07)} ${n(cell * 0.12)}"`;
  out += `<g class="pf-rim"${opts.plain ? '' : ` filter="url(#pfwob${id})"`}><g class="${
    wobble}">${
    circle(r - cell * FOLD_RIM / 2, `fill="none" stroke="${hue}" stroke-width="${n(cell * FOLD_RIM)}"`)}${
    circle(r - cell * 0.01, `fill="none" stroke="${hue}" stroke-width="${n(cell * 0.05)}" ${lumps}`)}</g></g>`;
  out += circle(r, `fill="none" stroke="${edge}" stroke-width="${n(cell * FOLD_KEYLINE)}" opacity="0.6"`);
  // The collar: who may move it. Above the pool, under any piece.
  if (movers.length) {
    const cr = cell * FOLD_COLLAR, w = cell * FOLD_COLLAR_WIDTH;
    out += `<g class="pf-collar" data-moved="${movers.join(' ')}">`;
    out += circle(cr, `fill="none" stroke="${edge}" stroke-width="${n(w + cell * 0.03)}"`);
    if (movers.includes(0) && movers.includes(1)) {
      const arc = (sweep, colour) => `<path d="M${n(cx - cr)} ${n(cy)}A${n(cr)} ${n(cr)} 0 0 ${sweep} ${
        n(cx + cr)} ${n(cy)}" fill="none" stroke="${colour}" stroke-width="${n(w)}"/>`;
      out += arc(0, PORTAL_SIDE[low]) + arc(1, PORTAL_SIDE[1 - low]);
    } else {
      out += circle(cr, `fill="none" stroke="${PORTAL_SIDE[movers.includes(0) ? 0 : 1]}" stroke-width="${n(w)}"`);
    }
    out += '</g>';
  }
  // The beak, on the rim, over the collar: past it at the full size, cut into it at the mid.
  if (aim !== null) {
    const beak = (cls, base, tip, half) => `<path class="${cls}" d="${portalBeakPath(cx, cy, aim,
      cell * base, cell * tip, cell * half)}" fill="${hue}" stroke="${edge}" stroke-width="${
      n(cell * 0.025)}" stroke-linejoin="round" paint-order="stroke"${dim}/>`;
    out += beak('pg-beak', FOLD_POOL - 0.07, FOLD_POOL + 0.06, 0.085);
    out += beak('pg-beak-cut', FOLD_POOL - 0.14, FOLD_POOL + 0.03, 0.09);
  }
  if (opts.numbered && !opts.occupied) {
    out += `<text class="pg-num" x="${n(cx)}" y="${n(cy)}" font-size="${n(cell * PORTAL_NUMBER)}"
      data-em="${PORTAL_NUMBER}" fill="${edge}" font-weight="700" text-anchor="middle"
      dominant-baseline="central">${end.pair}</text>`;
  }
  const label = opts.label ? ` role="img" aria-label="${String(opts.label).replace(/"/g, '&quot;')}"` : '';
  const said = aim !== null ? ` data-aim="${(aim * 180 / Math.PI).toFixed(1)}"` : '';
  return `<g class="fold" data-fold="${end.type}" data-portal="${end.id || end.pair + String(end.end)[0]}" data-pair="${
    end.pair}"${said}${opts.dim ? ' data-dim=""' : ''}${label}${opts.attrs ? ' ' + opts.attrs : ''}>${out}</g>`;
}

// The group's number with a piece standing in the fold: a small badge in the cell's
// top-right corner, drawn over the piece (§5.3).
function foldBadgeHtml(end, x, y, cell) {
  const bx = x + cell * FOLD_BADGE_AT[0], by = y + cell * FOLD_BADGE_AT[1];
  const n = v => v.toFixed(3);
  return `<g class="pf-badge" data-badge="${end.pair}" pointer-events="none"><circle class="pg-eye" cx="${n(bx)}" cy="${
    n(by)}" r="${n(cell * FOLD_BADGE)}" data-em="${FOLD_BADGE}" fill="var(--pair-${end.pair})"
    stroke="var(--portal-edge)" stroke-width="${n(cell * 0.025)}"/><text class="pg-num" x="${n(bx)}" y="${
    n(by)}" font-size="${n(cell * 0.2)}" data-em="0.2" fill="var(--portal-edge)" font-weight="700"
    text-anchor="middle" dominant-baseline="central">${end.pair}</text></g>`;
}

// The glow a pool spills onto the squares round it, laid over the board after its squares
// (a square drawn later would cover it): a ring in the hue, clear inside the rim so it never
// tints the piece standing in the fold, fading out past the cell's edge.
function foldGlowHtml(x, y, cell, pair) {
  const id = ++FOLD_POOLS;
  const cx = x + cell / 2, cy = y + cell / 2, n = v => v.toFixed(3);
  const from = FOLD_GLOW_IN / FOLD_GLOW_OUT, peak = (FOLD_POOL + 0.04) / FOLD_GLOW_OUT;
  return `<g class="pf-glow" data-glow="${pair}" pointer-events="none"><defs><radialGradient id="pfglow${id}">
    <stop offset="${n(from)}" stop-color="var(--pair-${pair})" stop-opacity="0"/>
    <stop offset="${n(peak)}" stop-color="var(--pair-${pair})" stop-opacity="0.55"/>
    <stop offset="1" stop-color="var(--pair-${pair})" stop-opacity="0"/></radialGradient></defs>
    <circle cx="${n(cx)}" cy="${n(cy)}" r="${n(cell * FOLD_GLOW_OUT)}" fill="url(#pfglow${id})"/></g>`;
}

// A fold end in words (§5.1's words, a fold's rule): the head as a momentum end's, and what
// it does. `there` names the end it leads to by square; `fromSquares` the ends leading here.
function foldWords(end, square, pairs, sideOf, there, fromSquares, nameOf) {
  const cap = text => text.charAt(0).toUpperCase() + text.slice(1);
  const open = end.open || [];
  const whose = open[0] && open[1] ? "both sides'"
    : open[0] ? `${sideOf(0)}'s` : open[1] ? `${sideOf(1)}'s` : null;
  const colour = pairs > 1 && PORTAL_PAIR_NAMES[end.pair - 1] ? ` (${PORTAL_PAIR_NAMES[end.pair - 1]})` : '';
  const head = `${cap(portalName(end, pairs))}${colour} on ${square}`;
  const movers = end.movedBy || [];
  const moved = !movers.length ? 'Nobody moves it.'
    : movers.length > 1 ? 'Either side may move it, as its turn, even under a piece.'
      : `Only ${sideOf(movers[0])} may move it, as a turn, even under a piece.`;
  // Schema 24 (`docs/archive/plan-portals-2.md` §5.3R): a piece that ends its move on a fold may
  // stay, or come out on the end it leads to; nothing else about a move changes.
  const from = (fromSquares || []).length ? ` Pieces come out here from ${portalList(fromSquares)}.` : '';
  const who = whose === "both sides'" ? 'a piece of either side' : `a piece of ${whose}`;
  const rule = whose && end.dest !== null
    ? `A fold: ${who} that ends its move on ${square} may stay there, or come out on ${
      there || 'the end it leads to'}, taking a piece there as any move does. A move that passes`
      + ` over it is not changed.${portalFilterWords(end, nameOf)}${from} ${moved}`
    : `A fold that leads nowhere: nothing goes through here.${from} ${moved}`;
  return { head: head, label: `${head}. ${rule}`, says: rule };
}

// The legend's (i) for a fold (Q843d; revised by Q867c, `docs/archive/plan-portals-2.md` §5.3R): the
// pools drawn as the board draws them, each with its sentence. First a pool with a piece
// standing in it, then a pair whose beaks and map arrows say where a piece comes out (no
// echo: the user found it clutter), then the bands (whose pieces may go through) and the
// collar (who may move it), each row changing one thing, as the vortex's key teaches its
// facts apart. An end's `at` is its cell along the row (its index where absent); an
// example's `arrows` are the group map's, `[from, to]` by those cells.
const FOLD_KEY_ROWS = [
  { head: 'A pool of light lying in the board', wide: true, examples: [
    { ends: [{ pair: 1, end: 'orange', movedBy: [], open: [true, true], aim: 0, piece: true }],
      says: () => 'A piece can stand in a fold, and is taken there as anywhere. A piece that'
        + ' ends its move on an empty fold may stay there, or come out on the end its beak points'
        + ' at, taking a piece there as any move does. A move that passes over it is not changed.' }] },
  { head: 'Its beak and the portal map: where a piece comes out', wide: true, examples: [
    { ends: [{ pair: 1, end: 'orange', movedBy: [], open: [true, true], aim: 0, at: 0 },
             { pair: 1, end: 'blue', movedBy: [], open: [true, true], aim: Math.PI, at: 2 }],
      arrows: [[0, 2], [2, 0]],
      says: () => 'The beak points at the end a piece comes out of, and the portal map draws'
        + ' the same as arrows. The same colour and number are one group: a pool is a fold'
        + ' whatever its colour.' }] },
  { head: 'Its bands: whose pieces may go through', examples: [
    { ends: [{ movedBy: [], open: [true, false] }], says: (s, c) => `${portalCap(c[0])} bands: ${s[0]}'s.` },
    { ends: [{ movedBy: [], open: [false, true] }], says: (s, c) => `${portalCap(c[1])} bands: ${s[1]}'s.` },
    { ends: [{ movedBy: [], open: [true, true] }], says: () => "One of each: both sides'." },
    { ends: [{ movedBy: [], open: [false, false] }], says: () => 'Plain: it leads nowhere.' }] },
  { head: 'Its collar: who may move it', examples: [
    { ends: [{ movedBy: [], open: [true, true] }], says: () => 'No collar: nobody.' },
    { ends: [{ movedBy: [0], open: [true, true] }], says: (s, c) => `A ${c[0]} collar: ${s[0]}.` },
    { ends: [{ movedBy: [1], open: [true, true] }], says: (s, c) => `A ${c[1]} collar: ${s[1]}.` },
    { ends: [{ movedBy: [0, 1], open: [true, true] }], says: () => 'Half and half: either side.' }] },
];

function foldKeyFigure(sides, icon) {
  const named = sides && sides.length === 2 ? sides : ['white', 'black'];
  const names = named.map(portalCap);
  const colours = named[0] === 'white' ? ['white', 'black'] : ['light', 'dark'];
  const cell = 48;
  // The piece standing in the pool: one of the game's own (`icon`, its first side's drawing,
  // as the board draws it), else the first side's disc.
  const piece = x => icon && icon.href
    ? `<image href="${String(icon.href).replace(/"/g, '&quot;')}" x="${x + cell * 0.06}" y="${cell * 0.06}" width="${cell * 0.88}"
        height="${cell * 0.88}" preserveAspectRatio="xMidYMid meet"/>`
    : `<circle cx="${x + cell / 2}" cy="${cell / 2}" r="${cell * 0.27}" fill="var(--p0)"
        stroke="var(--ink)" stroke-width="1.2"/>`;
  return `<div class="portal-figure fold-figure">${FOLD_KEY_ROWS.map(row => `<span class="portal-head">${
    row.head}</span><div class="portal-row${row.wide ? ' portal-pairs' : ''}">${
    row.examples.map(example => {
      const cellOf = (end, index) => (end.at === undefined ? index : end.at);
      const width = cell * (Math.max(...example.ends.map(cellOf)) + 1);
      const glyphs = example.ends.map((end, index) => {
        const x = cellOf(end, index) * cell;
        return foldGlyph({ pair: 1, end: 'orange', kind: 'fold', dest: 0, ...end }, x, 0, cell,
          { numbered: example.ends.length > 1, aim: end.aim, occupied: !!end.piece, plain: true })
          + (end.piece ? piece(x) + foldBadgeHtml({ pair: end.pair || 1 }, x, 0, cell) : '');
      }).join('') + (example.arrows || []).map(([from, to]) => portalArrowHtml(
        from * cell + cell / 2, cell / 2, to * cell + cell / 2, cell / 2, cell, 1, `${from}>${to}`)).join('');
      return `<div class="portal-example"><svg viewBox="0 0 ${width} ${cell}" width="${width}"
        height="${cell}" aria-hidden="true"><rect width="${width}" height="${cell}"
        fill="var(--light)"/>${glyphs}</svg><span>${example.says(names, colours)}</span></div>`;
    }).join('')}</div>`).join('')}</div>`;
}

// The ways out of a portal, by the direction the engine says the piece came out in
// (`exitDir`, `engine/compile.py`'s `PORTAL_DIRECTIONS` order: right, then round
// anticlockwise), as the board is drawn: turned round, its right is the reader's left.
const PORTAL_STEP_WORDS = ['right', 'up and right', 'up', 'up and left', 'left',
  'down and left', 'down', 'down and right'];
const PORTAL_STEP_ARROWS = ['→', '↗', '↑', '↖', '←', '↙',
  '↓', '↘'];
function portalStep(direction, flipped) {
  if (typeof direction !== 'number' || !(direction >= 0 && direction < 8)) return null;
  const seen = flipped ? (direction + 4) % 8 : direction;
  return { words: PORTAL_STEP_WORDS[seen], arrow: PORTAL_STEP_ARROWS[seen] };
}

// What a portal is called in a sentence: by its number where the game has several pairs.
// Never by which end of the pair it is: both ends are drawn alike, and the square says which
// one is meant (Q838c, `docs/archive/plan-portal-look.md` §5).
function portalName(end, pairs) {
  return pairs > 1 ? `portal ${end.pair}` : 'the portal';
}

// The pairs' colours as a person calls them, by number, `--pair-1` to `--pair-4`'s order
// (`web/page/tokens.py`). Said once, beside the number, as a friendly name; never the only
// way a pair is named, since the colours are hardest for the readers colour fails. Cyan and
// vermilion, not the two old end colours' names, which would say *which end* again.
const PORTAL_PAIR_NAMES = ['violet', 'yellow', 'cyan', 'vermilion'];

// The square a portal end of this type stands on in `board` (values as the core sends
// them, `nTypes` to an army), or -1. Where an end stands is the board's, as every piece's
// square is; this finds it and works nothing out.
function portalSquareIn(board, nTypes, type) {
  if (!board || !(nTypes > 0)) return -1;
  for (let square = 0; square < board.length; square++) {
    const value = board[square];
    if (value && (value - 1) % nTypes === type) return square;
  }
  return -1;
}

// One player's (or both players') passage through this end, as a clause. `there` names the
// partner by its square: *its partner on e4*, or *its partner* where the square is unknown.
function passageWords(kind, whose, there) {
  if (kind === 'both') return `${whose} pieces go in here or at ${there}, and come out beside the other`;
  if (kind === 'in') return `${whose} pieces go in here and come out beside ${there}`;
  if (kind === 'out') return `${whose} pieces come out beside this end, going in at ${there}`;
  return `${whose} pieces cannot go through: to them it is a solid block`;
}

// An end in words, for the label a screen reader reads, the tooltip on its square and the
// line under the board (D9: a person never has to decode the icon to know the rule).
// `square` is its name and `partnerSquare` its partner's (null where unknown);
// `sideOf(player)` the side's name as the ruleset says it. Returns `{head, label, says}`:
// which end it is (*Portal 1 (violet) on b3*, *The portal on e3*), the label (that and its
// rule), and the rule with how a piece comes out, for the line under the board after the
// head. The pair's colour is said once, in the head, and only where there are several pairs.
// `nameOf` names the pieces a filtered end lets in (`portalNamer`).
function portalWords(end, partner, square, pairs, sideOf, partnerSquare, fromSquares, nameOf) {
  const cap = text => text.charAt(0).toUpperCase() + text.slice(1);
  if (portalMapped(end, partner)) {
    return portalMapWords(end, square, pairs, sideOf, partnerSquare, fromSquares || [], nameOf);
  }
  const there = partnerSquare ? `its partner on ${partnerSquare}` : 'its partner';
  const first = portalPassage(end, partner, 0), second = portalPassage(end, partner, 1);
  const clauses = first === second ? [passageWords(first, "both sides'", there)]
    : [passageWords(first, `${sideOf(0)}'s`, there), passageWords(second, `${sideOf(1)}'s`, there)];
  const movers = end.movedBy || [];
  const moved = !movers.length ? 'Nobody moves it.'
    : movers.length > 1 ? 'Either side may move it, as its turn.'
      : `Only ${sideOf(movers[0])} may move it, as a turn.`;
  const colour = pairs > 1 && PORTAL_PAIR_NAMES[end.pair - 1]
    ? ` (${PORTAL_PAIR_NAMES[end.pair - 1]})` : '';
  const head = `${cap(portalName(end, pairs))}${colour} on ${square}`;
  const rule = `${clauses.map(cap).join('. ')}. ${moved}`;
  const through = first === 'closed' && second === 'closed' ? ''
    : ' A piece comes out one step beyond the other end, in the direction it was going.';
  const filter = portalFilterWords(end, nameOf);
  return { head: head, label: `${head}. ${rule}${filter}`, says: rule + filter + through };
}

// An end of a map in words (schema 21, `docs/archive/plan-portals-2.md` §5.1), by square: whose
// pieces go in here and which square they come out beside, or *Nothing goes in here*, and
// where pieces come out here from. `fromSquares` names the ends that lead here.
function portalMapWords(end, square, pairs, sideOf, partnerSquare, fromSquares, nameOf) {
  const cap = text => text.charAt(0).toUpperCase() + text.slice(1);
  const open = end.open || [];
  const whose = open[0] && open[1] ? "both sides'"
    : open[0] ? `${sideOf(0)}'s` : open[1] ? `${sideOf(1)}'s` : null;
  const beside = partnerSquare ? `beside ${partnerSquare}` : 'beside the portal it leads to';
  const from = fromSquares.length ? `pieces come out here from ${portalList(fromSquares)}` : '';
  const into = whose ? `${cap(whose)} pieces go in here and come out ${beside}`
    : `Nothing goes in here${from ? `: ${from}` : ''}`;
  const clauses = whose && from ? [into, cap(from)] : [into];
  const movers = end.movedBy || [];
  const moved = !movers.length ? 'Nobody moves it.'
    : movers.length > 1 ? 'Either side may move it, as its turn.'
      : `Only ${sideOf(movers[0])} may move it, as a turn.`;
  const colour = pairs > 1 && PORTAL_PAIR_NAMES[end.pair - 1]
    ? ` (${PORTAL_PAIR_NAMES[end.pair - 1]})` : '';
  const head = `${cap(portalName(end, pairs))}${colour} on ${square}`;
  const rule = `${clauses.join('. ')}.${portalFilterWords(end, nameOf)} ${moved}`;
  const through = whose
    ? ' A piece comes out one step beyond the portal it leads to, in the direction it was going.'
    : '';
  return { head: head, label: `${head}. ${rule}`, says: rule + through };
}

// The status line's words when the end on `square` has just been moved and may not go
// straight back (schema 20's `no_straight_back`): the core's bar (`portalBar`, `{standing,
// left}` off the state) names the end and the square it left, and the engine has already
// left that move out; this only says so. `name(square)` names a square. Empty otherwise.
function portalBarWords(bar, square, end, pairs, name) {
  if (!bar || !end || bar.standing !== square || !(bar.left >= 0)) return '';
  const said = portalName(end, pairs);
  return `${said.charAt(0).toUpperCase() + said.slice(1)} has just moved and may not go `
    + `straight back to ${name(bar.left)} this turn`;
}

// The legend's (i) for a portal (Q838c, §5; two rows since Q838f): the ends drawn as the
// board draws them, each with its sentence, so the vortex is learnt once from pictures
// rather than from words. The two facts are taught apart, as two rows that each change one
// thing: what an end stands on (who may move it), every stand with the same arms; and its
// arms (whose pieces go in), every one on the same sunk well. The cold read
// (`docs/reference/portal-look-cold-read.md` §4) took the old key's one picture of white arms
// on a white token for *white arms mean White's token*, and guessed at the stands it never
// drew. Last, two ends of one pair. `sides` are the two sides' names (`viz.pieces.side_names`:
// white and black, or first and second for a set told apart by turning). Returns markup.
// The `end` fields are the payload's names for the two ends of a pair, never words.
const portalCap = text => String(text).charAt(0).toUpperCase() + String(text).slice(1);
const PORTAL_KEY_ROWS = [
  { head: 'What it stands on: who may move it', examples: [
    { ends: [{ movedBy: [0], open: [true, true] }],
      says: (s, c) => `On a ${c[0]} token: ${s[0]} moves it.` },
    { ends: [{ movedBy: [1], open: [true, true] }],
      says: (s, c) => `On a ${c[1]} token: ${s[1]} moves it.` },
    { ends: [{ movedBy: [0, 1], open: [true, true] }],
      says: () => 'Half and half: either side moves it.' },
    { ends: [{ movedBy: [], open: [true, true] }],
      says: () => 'Sunk into the board: nobody moves it.' }] },
  { head: 'Its arms: whose pieces go in', examples: [
    { ends: [{ movedBy: [], open: [true, false] }],
      says: (s, c) => `${portalCap(c[0])} arms: ${s[0]}'s pieces go in.` },
    { ends: [{ movedBy: [], open: [false, true] }],
      says: (s, c) => `${portalCap(c[1])} arms: ${s[1]}'s pieces go in.` },
    { ends: [{ movedBy: [], open: [true, true] }],
      says: () => "One of each: both sides' pieces go in." },
    { ends: [{ movedBy: [], open: [false, false] }],
      says: () => 'No arms: nobody goes in. It is a way out.' }] },
  // Q843b (schema 21, `docs/archive/plan-portals-2.md` §5.1, §5.2): the beak, taught on a pair whose
  // beaks point at each other and on a funnel whose beaks all point at its way out; and the
  // dashed arms of an end that lets only some pieces in.
  { head: 'Its colour, number and beak: where it leads', wide: true, examples: [
    { ends: [{ pair: 2, end: 'orange', movedBy: [], open: [true, true], aim: 0 },
             { pair: 2, end: 'blue', movedBy: [], open: [true, true], aim: Math.PI }],
      says: () => 'The same colour and number are one group. The beak points where you come'
        + ' out: in at one, out beside the end it points at.' },
    { ends: [{ pair: 3, end: 'a', movedBy: [], open: [true, true], aim: 0 },
             { pair: 3, end: 'b', movedBy: [], open: [true, true], aim: 0 },
             { pair: 3, end: 'c', movedBy: [], open: [false, false] }],
      says: () => 'Several beaks may point at one end: pieces come out beside it, and it has'
        + ' no arms, since nothing goes in there.' }] },
  { head: 'Dashed arms: only some pieces', examples: [
    { ends: [{ pair: 1, end: 'orange', movedBy: [], open: [true, true], filtered: true }],
      says: () => 'Dashed arms: only some pieces may go in. The words under the board say'
        + ' which, and when you pick a piece the doors it may not use go dim.' }] },
];

function portalKeyFigure(sides) {
  const named = sides && sides.length === 2 ? sides : ['white', 'black'];
  const plain = named[0] === 'white';
  const names = named.map(portalCap);
  const colours = plain ? ['white', 'black'] : ['light', 'dark'];
  const cell = 48;
  return `<div class="portal-figure">${PORTAL_KEY_ROWS.map(row => `<span class="portal-head">${
    row.head}</span><div class="portal-row${row.wide || row.examples.length === 1 ? ' portal-pairs' : ''}">${
    row.examples.map(example => {
      const width = cell * example.ends.length;
      const glyphs = example.ends.map((end, at) => portalGlyph(
        { pair: 1, end: 'orange', ...end }, at * cell, 0, cell,
        { numbered: example.ends.length > 1, aim: end.aim })).join('');
      return `<div class="portal-example"><svg viewBox="0 0 ${width} ${cell}" width="${width}"
        height="${cell}" aria-hidden="true"><rect width="${width}" height="${cell}"
        fill="var(--light)"/>${glyphs}</svg><span>${example.says(names, colours)}</span></div>`;
    }).join('')}</div>`).join('')}</div>`;
}

// The kinds a board carries beyond its square marks, for the legend line: a wall where
// the rules put one (`obstacles`), and a portal where the game has them.
function boardKinds(walls, portals) {
  // Schema 22: a momentum end is a *portal*, a fold a *fold portal*, each named where the
  // game has one (a payload from before schema 22 has no `kind`: every end is momentum).
  const ends = (portals && portals.ends) || [];
  return [...(walls && walls.length ? ['wall'] : []),
          ...(portals && (!ends.length || !ends.every(isFold)) ? ['portal'] : []),
          ...(ends.some(isFold) ? ['fold'] : [])];
}

// The copy of `square` drawn nearest the cell (u, v): the square itself, or one of its
// ghosts round a wrapping board. On a view slid across the seam an exit beside a portal at
// the outline's edge is drawn beside it only as a ghost (N6), and that is the copy to light.
function nearestCopy(square, near, view) {
  const own = squareCell(square, view);
  let best = { u: own[0], v: own[1], depth: 0 };
  const far = c => Math.hypot(c.u - near[0], c.v - near[1]);
  for (const ghost of ghostCells(view)) {
    if (ghost.square === square && far(ghost) < far(best)) best = ghost;
  }
  return best;
}

// ---------------------------------------------------------------- a ply is beats
//
// **One submitted move can make two board changes.** A ruleset with `rules.npcs` runs a
// wild-piece phase *after* the mover's move, inside the same `make`, and that phase may
// capture with no player piece moving to the square. A surface that shows the ply as
// one transition has nothing to show as the taker, which is exactly what a wild piece
// "captured by nothing" is.
//
// So a ply is a **sequence of beats**, and this splits one into them:
//
//   [{kind: 'move', board}, {kind: 'phase', board, steps, walkers}]
//
// `board` is the position at the end of that beat, so the beats are positions a viewer
// can be *shown* one at a time and not only a script for an animation. There is one
// beat when no wild piece moved, which is every ruleset before schema 9.
//
// Here rather than in either renderer because it has one right answer: the play board
// animates the two beats and steps through them, the offline replay steps through them,
// and a replay that split a ply differently from the board would be a second account of
// the same game.
//
// **The steps come from the core.** `last.phase` is `Game::npc_log` as the engine wrote
// it -- which piece walked where, and the board byte it ate when it got there. This reads
// the ply rather than reconstructing it, so the two things a diff could only guess at
// come out right: a pawn eaten on the square it promoted onto is the piece it *became*,
// and a castling rook eaten in transit is a rook being taken rather than a quiet
// disappearance.
//
// `walkers` is the board value of the piece that took each step, in the steps' order,
// which the log does not carry and the walk back finds for free: it is what stands on
// the step's destination at the moment the step is undone.
function plyBeats(after, last) {
  const whole = [{ kind: 'move', board: after, steps: null }];
  const steps = (last && last.phase) || [];
  if (!steps.length) return whole;

  // The position between the beats: the phase, walked back. Newest step first, and the
  // wild piece's own value read off the square it stands on now -- `Game::unmake`'s
  // order exactly, which is the one that stays right if a step ever lands where another
  // started.
  const mid = after.slice();
  const walkers = new Array(steps.length);
  for (let i = steps.length - 1; i >= 0; i--) {
    const step = steps[i];
    walkers[i] = mid[step.to];
    mid[step.from] = mid[step.to];
    mid[step.to] = step.eaten;
  }
  return [{ kind: 'move', board: mid, steps: null },
          { kind: 'phase', board: after, steps: steps, walkers: walkers }];
}

// What a ply's wild-piece phase did, in words, for the line a move list gives that beat.
// It reads the ply's own account -- the steps the core logged and the walkers `plyBeats`
// found -- and names pieces through `nameOf`, a board value to the spec's name for it,
// which each renderer knows its own way. Nothing is worked out: the squares are the
// engine's names (`squareName`) and the order is the order the core walked them in.
//
//   and the wolf on c6 steps to c5
//   and the wolf on c4 takes the bishop on c5
//   and the wolf on a4 steps to b5, h4 to g5, a5 to b4 and h5 to g4
//
// The last form is a run of quiet steps by pieces of one name, which is most plies of a
// game with several wild pieces on patrol: the name once, then the squares. A capture is
// always said in full, because it is the step a reader is looking for. Plain text, for
// the caller to escape.
function phaseWords(steps, walkers, nameOf, W) {
  const at = square => squareName(square % W, Math.floor(square / W));
  const clauses = [];
  let walking = null;       // the name of the piece whose quiet steps are being listed
  steps.forEach((step, i) => {
    const who = (walkers && walkers[i] && nameOf(walkers[i])) || 'wild piece';
    if (step.eaten) {
      clauses.push('the ' + who + ' on ' + at(step.from) + ' takes the '
                   + (nameOf(step.eaten) || 'piece') + ' on ' + at(step.to));
      walking = null;
    } else if (who === walking) {
      clauses.push(at(step.from) + ' to ' + at(step.to));
    } else {
      clauses.push('the ' + who + ' on ' + at(step.from) + ' steps to ' + at(step.to));
      walking = who;
    }
  });
  const last = clauses.pop();
  return 'and ' + (clauses.length ? clauses.join(', ') + ' and ' + last : last);
}
