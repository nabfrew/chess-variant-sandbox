// A still board: a spec that has never been played, drawn once.
//
// Deliberately not ``drawBoard``, which draws the position of a game in progress and reads
// ``SESSION``. This draws a payload -- the ``preview`` block ``designing.check`` sends, or
// anything shaped like it -- and returns a string. It keeps no state, asks no server and
// names no piece, so any view can call it: the designer's workbench board does, and stage 5
// of ``docs/plan-designer-ux.md`` puts a played game on the same drawing.
//
// It shares ``squareXY`` with the play board, because there is one answer to where a square
// is drawn and a second copy of that arithmetic is how a board comes to half agree with
// itself. Everything a rule singles out -- the goal ring, the promotion band, the seam --
// is the play board's own ``squareMarks``, ``squareBadge`` and ``barrierHtml`` read off the
// compiled rules the payload carries.
//
// Every square is a ``data-square`` group, which is what a click on it is read off.

// ------------------------------------------------------------------ a still board

// `map` marks the engine's answer about one piece: a dot where it may move, a ring where
// it may take, and the square it stands on tinted. `drawn` is a drawing in progress: its
// squares are tinted under the engine's marks, so where the two agree a dot sits in a
// tinted square and where they differ it does not.
let STILL_TAGS = 0;

// `bench` is the designer's workbench (P3 of `docs/plan-portals.md`), and absent everywhere
// else: what its board says over the position about the portals being put down and the
// keyboard's place on it. Every square in it is one the designer's own state names --
// `ring` the selected pair's ends, `lift` an end picked up to move, `dim` the other pairs'
// ends, `pending` the orange end of a pair still being put down ({square, pair}), `flash`
// a square a refused click was on, `focus` the keyboard's square, `link` the selected
// pair's two ends -- and the board marks them; it works out nothing about a move.
function boardPreview(preview, map, drawn, bench) {
  if (!preview) return '';
  const W = preview.width, H = preview.height;
  const cell = 100 / Math.max(W, H);
  const view = flatView(W, H, FLIPPED, preview.lattice, boardCut(preview.absent, preview.wrap));
  const walls = new Set(preview.obstacles || []);
  const moves = new Set((map && map.moves) || []);
  const takes = new Set((map && map.takes) || []);
  const drewMoves = new Set((drawn && drawn.moves) || []);
  const drewTakes = new Set((drawn && drawn.takes) || []);
  // The reader's own toggle, shared with the play board and the replay through one
  // stored key. The preview offers no control of its own -- a still of a draft is not
  // where a promotion band gets put away -- but it honours the one already set.
  // A thumbnail (`plain`) draws none: a card has no legend line to name a mark by (S3),
  // and a glance at a game is its board and its army.
  const marks = preview.plain ? {} : squareMarks(preview, { zone: !zonesShown() });
  // Its own name for the hatch pattern, so two stills on one page never share one.
  const tag = 'still' + (++STILL_TAGS);
  let out = markDefs(marks, tag);
  // Its wood is hashed from the game's hash where the caller gives one (a library card's
  // thumbnail), else from its shape: a draft's hash moves with every edit, and its wood
  // should not.
  const identity = preview.hash || `${W}x${H}:${preview.lattice || 'square'}`;
  const wood = woodGrain(tag, identity, cell);
  // Schema 19 (D5): the engine's transits for the piece asked about -- the squares it comes
  // out on through a portal, in the colour of the end it goes in by, and the ends it can go
  // in by -- and the squares it also reaches straight. Read off the map; nothing here
  // works out where a piece comes out.
  const through = new Map(), entries = new Set();
  const straight = new Set((map && map.direct) || []);
  for (const transit of (map && map.transits) || []) {
    const end = stillPortalAt(preview, transit.portal);
    through.set(transit.to, end && end.end === 'blue' ? 'var(--portal-b)' : 'var(--portal-a)');
    entries.add(transit.portal);
  }
  const marked = bench || {};
  const has = (set, square) => !!set && set.has(square);
  // The seam, where the rules put one: its arrowheads open the frame (below).
  const seams = seamAxes(preview.wrap);
  const mx = seams.files ? cell * (SEAM_ARROW + 0.05) : 0;
  const my = seams.ranks ? cell * (SEAM_ARROW + 0.05) : 0;
  // Q809: the designer's bench names its squares as the play board does -- outside the
  // board where its drawing ends at its own edge (`edgeNamesHtml`, as `drawCoords` sets
  // them), on the edge squares where a seam's arrows stand round it -- because its
  // sentences and the squares a person is asked to click are named. A thumbnail and the
  // game page's start position name none.
  const named = !!bench && !preview.plain;
  const outside = named && !mx && !my && labelsOutside(view);
  for (let rank = H - 1; rank >= 0; rank--) {
    for (let file = 0; file < W; file++) {
      const square = rank * W + file;
      if (cellAbsent(square, view)) continue;
      const [x, y] = squareXY(square, cell, view);
      const shade = cellShade(file, rank, view);
      out += `<g data-square="${square}">`
        + cellShape(x, y, cell, view, `fill="${shade}"`) + wood.cell(x, y, view, square, shade);
      if (walls.has(square)) {
        out += cellShape(x, y, cell, view, 'fill="var(--wall)"');
      }
      if (drewMoves.has(square)) {
        out += cellShape(x, y, cell, view, 'fill="var(--hl-sel)" opacity="0.28"');
      }
      if (drewTakes.has(square)) {
        out += cellShape(x, y, cell, view, `fill="none" stroke="var(--bad)" stroke-width="${
          cell * 0.05}" stroke-dasharray="${cell * 0.1}" opacity="0.8"`, 0.08);
      }
      // Under the piece, for the play board's reason: these squares are singled out for
      // being arrived on, so whatever is standing there has to stay what you see first.
      if (marks[square]) out += squareBadge(x, y, cell, marks[square], view, tag);
      const value = preview.board[square];
      if (value) {
        const stride = preview.sideStride || preview.nTypes;
        const type = (value - 1) % preview.nTypes, player = Math.floor((value - 1) / stride);
        const facing = Math.floor(((value - 1) % stride) / preview.nTypes);
        // Never an army's icon for a wild piece (owner 2, schema 9): an icon pair
        // has one drawing per player, and the third owner is neither.
        const icon = player > 1 ? null : ((preview.icons || [])[type] || [])[player];
        // Schema 19: a portal end, drawn as the play board draws it (`portalGlyph`), its
        // passage shown for the side that moves first from here; a thumbnail draws the
        // glyph's least, rim and number (D9).
        const end = preview.portals ? preview.portals.ends.find(one => one.type === type) : null;
        if (end) {
          const partner = portalPartner(preview.portals, end);
          const pairs = portalPairs(preview.portals);
          const name = squareName(file, rank);
          const side = player => ((preview.sides || [])[player]) || `player ${player}`;
          // The designer numbers even a lone pair, where a second is about to be added (D3).
          const numbered = pairs > 1 || !!bench;
          const glyph = preview.plain
            ? `<g class="pg-small">${portalGlyph(end, partner, null, x, y, cell,
                { numbered })}</g>`
            : portalGlyph(end, partner, preview.toMove === undefined ? null : preview.toMove,
                x, y, cell, { numbered,
                  label: portalWords(end, partner, name, pairs, side).label });
          out += has(marked.dim, square) ? `<g opacity="0.4">${glyph}</g>` : glyph;
        } else if (icon) {
          // Turned only where the piece has a facing; a single-colour set's 180 is
          // ownership. Same rule as the play board, read off the same list.
          const heading = (preview.directional || [])[type]
            ? (preview.facingAngles || [])[player] : null;
          const angle = heading ? heading[facing] || 0 : (icon.rotate ? 180 : 0);
          out += `<image href="${esc(icon.href)}" x="${x + cell * 0.06}" y="${y + cell * 0.06}"
            width="${cell * 0.88}" height="${cell * 0.88}" preserveAspectRatio="xMidYMid meet"${
            angle ? ` transform="rotate(${angle} ${x + cell / 2} ${y + cell / 2})"` : ''}/>`;
        } else {
          out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.34}"
            fill="${player === 0 ? 'var(--p0)' : player === 1 ? 'var(--p1)' : 'var(--p2)'}"
            stroke="var(--ink)" stroke-width="${cell * 0.025}"/>
            <text x="${x + cell / 2}" y="${y + cell / 2}" font-size="${cell * 0.4}"
              text-anchor="middle" dominant-baseline="central" font-weight="600"
              fill="${player === 0 ? 'var(--p0-ink)'
                    : player === 1 ? 'var(--p1-ink)' : 'var(--p2-ink)'}"
              >${esc(preview.pieceIds[type])}</text>`;
        }
      }
      if (map && square === map.from) {
        out += cellShape(x, y, cell, view, 'fill="var(--hl-sel)" opacity="0.5"');
      } else {
        const way = through.get(square);
        if (way && !straight.has(square)) {
          // Reached only through a portal: the dot in the entry end's colour, edged so it
          // holds on both woods, as the play board draws it.
          out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.14}"
            fill="${way}" stroke="var(--portal-edge)" stroke-width="${cell * 0.03}"
            data-through="1"/>`;
        } else if (moves.has(square)) {
          out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.14}"
            fill="var(--dot)" opacity="0.9"/>`;
        }
        if (takes.has(square)) {
          out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.38}"
            fill="none" stroke="var(--bad)" stroke-width="${cell * 0.06}" opacity="0.85"/>`;
        }
        if (way && straight.has(square)) {
          // Reached straight and through a portal too: a bead of the entry's colour on
          // the square's corner.
          out += `<circle cx="${x + cell * 0.8}" cy="${y + cell * 0.2}" r="${cell * 0.09}"
            fill="${way}" stroke="var(--portal-edge)" stroke-width="${cell * 0.025}"
            data-through="1"/>`;
        }
      }
      if (entries.has(square)) {
        // An end the piece can go in by: a dashed ring, as on the play board.
        out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.47}" fill="none"
          stroke="var(--dot)" stroke-width="${cell * 0.035}" stroke-dasharray="${cell * 0.08}
          ${cell * 0.06}" opacity="0.8" data-entry="${square}"/>`;
      }
      out += benchMarksHtml(marked, square, x, y, cell, view, preview);
      if (named && !outside) out += coordsHtml(square, x, y, cell, view);
      out += '</g>';
    }
  }
  // Over the squares: the selected pair's link, and each transit's two legs, drawn hidden
  // and shown while the square it comes out on, or the end it goes in by, is hovered (D5).
  out += benchLinesHtml(marked, map, preview, cell, view);
  // The seam, where the rules put one. A preview is one repeat drawn in its own frame,
  // so the seam falls on the low edge and its copy on the high one -- the same line
  // twice. The frame opens by an arrowhead's reach on that axis (`mx`, `my`): the markers
  // point out of the board and there are no ghost rings here to hold them.
  // A hexagon's sides every player crosses both ways are drawn as repeat lines, since a
  // still has no ghost copy beyond them to say so.
  if (view.cut) out += outlineHtml(cell, view, preview.wrap);
  out += barrierHtml(cell, view, preview.wrap);
  const box = view.hex ? viewBox(view, cell) : `${-mx} ${-my} ${W * cell + 2 * mx} ${H * cell + 2 * my}`;
  const svg = `<svg viewBox="${box}" style="width:100%; height:auto; display:block">${
    wood.defs()}${out}</svg>`;
  if (!outside) {
    return `<div class="board-frame" style="${woodFrameStyle(identity)}">${svg}</div>${
      handStripHtml(preview)}`;
  }
  return `<div class="still-named"><div class="board-frame" style="${
    woodFrameStyle(identity)}"><div class="board-box">${svg}<div class="coords${
    view.hex ? ' on-wood' : ''}" aria-hidden="true">${edgeNamesHtml(cell, view, box)}</div></div></div></div>${
    handStripHtml(preview)}`;
}

// The squares' names set outside a board, one `.co` each beside its own edge cell, placed
// as a share of the drawing's `box` (its viewBox) and set off outwards by `--co-out`. The
// play board's `drawCoords` and the designer's bench both draw them here, so the two name
// their squares alike.
function edgeNamesHtml(cell, view, box) {
  const frame = String(box).split(/\s+/).map(Number);
  return edgeLabels(cell, view).map(label => `<span class="co" style="left:${
    ((label.x - frame[0]) / frame[2] * 100).toFixed(3)}%;top:${
    ((label.y - frame[1]) / frame[3] * 100).toFixed(3)}%;--dx:${label.dx.toFixed(3)};--dy:${
    label.dy.toFixed(3)}">${esc(label.text)}</span>`).join('');
}

// The portal end standing on a square of a still, from its board byte and the compiled
// payload (`portal_payload`), or null.
function stillPortalAt(preview, square) {
  if (!preview || !preview.portals || square === undefined || square === null) return null;
  const value = preview.board[square];
  if (!value) return null;
  const type = (value - 1) % preview.nTypes;
  return preview.portals.ends.find(end => end.type === type) || null;
}

// The workbench's marks on one square (`boardPreview`'s `bench`): the selected pair's
// ends ringed, the end picked up to move ringed dashed, the orange end of a pair being put
// down, a refused click's square outlined once, and the keyboard's square.
function benchMarksHtml(bench, square, x, y, cell, view, preview) {
  let out = '';
  const cx = x + cell / 2, cy = y + cell / 2;
  for (const pending of bench.pending || []) {
    if (pending.square !== square) continue;
    out += portalGlyph({ pair: pending.pair, end: pending.end, movedBy: [], open: [true, true] },
      null, null, x, y, cell, { numbered: true, label: `pair ${pending.pair}, ${pending.end} end` });
  }
  if (bench.ring && bench.ring.has(square)) {
    out += `<circle cx="${cx}" cy="${cy}" r="${cell * 0.48}" fill="none" stroke="var(--hl-sel)"
      stroke-width="${cell * 0.035}" data-picked="${square}"/>`;
  }
  if (bench.lift === square) {
    out += `<circle cx="${cx}" cy="${cy}" r="${cell * 0.47}" fill="none" stroke="var(--portal-edge)"
      stroke-width="${cell * 0.05}" stroke-dasharray="${cell * 0.1} ${cell * 0.07}"
      data-lifted="${square}"/>`;
  }
  if (bench.flash === square) {
    out += cellShape(x, y, cell, view, `class="benchFlash" fill="none" stroke="var(--portal-edge)"
      stroke-width="${cell * 0.07}"`, 0.05);
  }
  if (bench.focus === square) {
    out += cellShape(x, y, cell, view, `fill="none" stroke="var(--focus)"
      stroke-width="${cell * 0.06}" data-kbfocus="${square}"`, 0.04);
  }
  return out;
}

// What the workbench draws across squares: the selected pair's two ends joined by a faint
// line (D2, D3), and every transit's two legs -- the piece's own way to the end it goes in
// by, then from the other end to the square it comes out on -- each along the mover's own
// step round a seam, hidden until its square or its end is hovered (D5).
function benchLinesHtml(bench, map, preview, cell, view) {
  const W = preview.width, H = preview.height;
  const centre = square => {
    const xy = squareXY(square, cell, view);
    return [xy[0] + cell / 2, xy[1] + cell / 2];
  };
  const allow = wrapAllow(preview.wrap, 0);
  const leg = (from, to, trimA, trimB, attrs) => {
    const [bx, by] = centre(to);
    const [dx, dy] = stepXY(wrapStep(from, to, W, H, allow), cell, view);
    return trimmedLine(bx - dx, by - dy, bx, by, trimA, trimB, cell, attrs);
  };
  let out = '';
  if (bench && bench.link) {
    const [a, b] = bench.link;
    const [ax, ay] = centre(a), [bx, by] = centre(b);
    out += `<g data-link="pair" pointer-events="none">${trimmedLine(ax, ay, bx, by, 0.46, 0.46,
      cell, `stroke="var(--portal-edge)" stroke-width="${cell * 0.04}" stroke-dasharray="${
      cell * 0.12} ${cell * 0.08}" stroke-linecap="round" opacity="0.6"`)}</g>`;
  }
  if (map && map.from >= 0) {
    const cased = `stroke="var(--portal-edge)" stroke-width="${cell * 0.12}"
      stroke-linecap="round" opacity="0.55"`;
    const lit = `stroke="var(--hl-last)" stroke-width="${cell * 0.065}" stroke-linecap="round"`;
    for (const transit of map.transits || []) {
      const both = (from, to, a, b) => leg(from, to, a, b, cased) + leg(from, to, a, b, lit);
      out += `<g class="benchLeg" data-exit="${transit.to}" data-entry="${transit.portal}"
        pointer-events="none">${both(map.from, transit.portal, 0.1, 0.45)}${
        both(transit.portalOut, transit.to, 0.45, 0.3)}</g>`;
    }
  }
  return out;
}

// What each player holds off the board. A hand is a count per piece type, the way the
// state carries it, so index 2 holding 3 means three of piece type 2.
//
// Drawn whenever the rules give the game a hand at all, empty or not: a game whose hands
// fill from captures starts with nothing in them, and a board with no strip under it is
// a board that says there is no hand -- which is the wrong thing to say about it.
function handStripHtml(preview) {
  if (!preview.handsUsed) return '';
  const sides = preview.sides || ['first', 'second'];
  const rows = (preview.hands || []).map((hand, player) => {
    const held = (hand || []).map((count, type) => [type, count])
      .filter(([, count]) => count > 0);
    if (!held.length) return '';
    return `<div class="row small stillHand"><span class="muted">${
      esc(sides[player])} holds</span>${held.map(([type, count]) =>
        `<span class="chip">${esc(preview.pieceIds[type])} &times;${count}</span>`).join('')}
      </div>`;
  }).join('');
  if (rows) return rows;
  return `<div class="row small stillHand"><span class="muted">${
    preview.handsFromCaptures
      ? 'Both hands are empty. A piece that is taken goes into its taker\'s hand and can'
        + ' be played back onto the board.'
      : 'Both hands are empty at the start.'}</span></div>`;
}

// A board with one piece on it and nothing else, for showing what a descriptor means.
// The payload is the opening position's, with its board replaced -- so the size, the
// lattice, the obstacles, the icons and the marks a rule puts on a square all stay the
// preview's own, and only what is standing where changes.
//
// Schema 19 (N5): the portals stay where the setup puts them, as the walls do, so the
// engine's answer for the lone piece -- which keeps them too (`designing.move_map`) -- is
// drawn among them, transits and all.
function loneBoard(preview, square, value) {
  return Object.assign({}, preview, {
    board: preview.board.map((held, at) => (at === square ? value
      : stillPortalAt(preview, at) ? held : 0)),
    // One piece on an empty board is a diagram of what a descriptor means, and nobody
    // is holding anything in it.
    handsUsed: false,
  });
}
