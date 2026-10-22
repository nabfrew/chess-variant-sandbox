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

// `bench` is the designer's workbench (P3 of `docs/archive/plan-portals.md`), and absent everywhere
// else: what its board says over the position about the portals being put down and the
// keyboard's place on it. Every square in it is one the designer's own state names --
// `ring` the selected pair's ends, `lift` an end picked up to move, `dim` the other pairs'
// ends, `pending` the first end of a pair still being put down ({square, pair}), `flash`
// a square a refused click was on, `focus` the keyboard's square, `link` the selected
// pair's two ends -- and the board marks them; it works out nothing about a move.
function boardPreview(preview, map, drawn, bench) {
  if (!preview) return '';
  const W = preview.width, H = preview.height;
  const cell = 100 / Math.max(W, H);
  const view = flatView(W, H, FLIPPED, preview.lattice, boardCut(preview.absent, preview.wrap));
  const walls = new Set(preview.obstacles || []);
  // Schema 30 (Q999g): `[[square, bits], ...]` as the core reports them for this position.
  const effects = new Map(preview.plain ? [] : effectPairs(preview.effects));
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
  // out on through a portal, in the colour of the pair it goes in by, and the ends it can go
  // in by -- and the squares it also reaches straight. Read off the map; nothing here
  // works out where a piece comes out.
  const through = new Map(), entries = new Set();
  const straight = new Set((map && map.direct) || []);
  for (const transit of (map && map.transits) || []) {
    // Schema 24 (`docs/archive/plan-portals-2.md` §5.3R): through a fold, the move ends on the far
    // end, which takes the ordinary dot (it is in the map's `moves`); a fold is no way in to
    // tap, so it is not ringed.
    if (transit.foldIn !== undefined) continue;
    const end = stillPortalAt(preview, transit.portal);
    through.set(transit.to, end ? `var(--pair-${end.pair})` : 'var(--dot)');
    entries.add(transit.portal);
  }
  // Schema 22 (Q843d, `docs/archive/plan-portals-2.md` §5.3): the folds, which are no board byte,
  // where the payload says they lie (`fold_payload`). Nothing is drawn on the far end
  // (§5.3R: the echo went; the beak says where a fold leads).
  const folds = new Map((preview.folds || []).map(fold => [fold.square, fold]));
  const groups = preview.portals ? portalPairs(preview.portals) : 0;
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
      // A fold, under any piece standing in it, as the play board draws it; still, and plain
      // on a thumbnail. Only the designer's bench turns it.
      const fold = folds.get(square);
      const foldEnd = fold && preview.portals ? preview.portals.ends.find(one => one.type === fold.type) : null;
      if (foldEnd) {
        const to = portalPartner(preview.portals, foldEnd);
        const there = to ? ((preview.folds || []).find(one => one.type === to.type) || {}).square : undefined;
        const glyph = foldGlyph(foldEnd, x, y, cell, { numbered: groups > 1 || !!bench,
          occupied: !!value, flipped: view.flipped, plain: !!preview.plain, idle: !!bench,
          aim: preview.plain ? null : portalAimFor(foldEnd, there === undefined ? -1 : there, x, y, cell, view),
          label: preview.plain ? '' : stillFoldLabel(preview, foldEnd, square, there) });
        out += has(marked.dim, square) ? `<g opacity="0.4">${glyph}</g>` : glyph;
      }
      if (value) {
        const stride = preview.sideStride || preview.nTypes;
        const type = (value - 1) % preview.nTypes, player = Math.floor((value - 1) / stride);
        // Schema 19: a portal end, drawn as the play board draws it (`portalGlyph`); a
        // thumbnail draws the glyph's least, its arms as wedges (D9). Only the designer's
        // bench, a board being worked on, turns its arms idly: a still stands still, so a
        // page of cards is never busy.
        const end = preview.portals ? preview.portals.ends.find(one => one.type === type) : null;
        if (end) {
          const partner = portalPartner(preview.portals, end);
          const pairs = portalPairs(preview.portals);
          const name = squareName(file, rank);
          const side = player => ((preview.sides || [])[player]) || `player ${player}`;
          const there = partner ? portalSquareIn(preview.board, preview.nTypes, partner.type) : -1;
          const thereName = there >= 0 ? squareName(there % W, Math.floor(there / W)) : null;
          const from = portalSources(preview.portals, end)
            .map(source => portalSquareIn(preview.board, preview.nTypes, source.type))
            .filter(at => at >= 0).map(at => squareName(at % W, Math.floor(at / W)));
          // The designer numbers even a lone pair, where a second is about to be added (D3).
          const numbered = pairs > 1 || !!bench;
          // Schema 21 (Q843b): the beak, aimed at where the end it leads to stands in this
          // still, so it re-aims as the designer moves an end (no beak on a thumbnail: it is
          // drawn too small to hold one).
          const aim = portalAimFor(end, there, x, y, cell, view);
          const glyph = preview.plain
            ? `<g class="pg-small">${portalGlyph(end, x, y, cell,
                { numbered, flipped: view.flipped })}</g>`
            : portalGlyph(end, x, y, cell, { numbered, flipped: view.flipped, aim,
                idle: !!bench, label: portalWords(end, partner, name, pairs, side, thereName, from,
                  portalNamer(preview.pieceIds, preview.pieceNames)).label });
          out += has(marked.dim, square) ? `<g opacity="0.4">${glyph}</g>` : glyph;
        } else {
          out += stillPieceHtml(preview, value, x, y, cell);
          if (foldEnd && (groups > 1 || !!bench)) out += foldBadgeHtml(foldEnd, x, y, cell);
        }
      }
      // Schema 30 (Q999g): the badges of the effects a line of sight puts on the piece, the
      // play board's own (`effectBadges`), off the core's answer for this position (the
      // payload's `effects`). Not on a thumbnail, which has no legend to name them by.
      if (value && effects.has(square)) {
        const stride = preview.sideStride || preview.nTypes;
        out += effectBadges(effects.get(square), x, y, cell, Math.floor((value - 1) / stride));
      }
      if (map && square === map.from) {
        out += cellShape(x, y, cell, view, 'fill="var(--hl-sel)" opacity="0.5"');
      } else {
        const way = through.get(square);
        if (way && !straight.has(square)) {
          // Reached only through a portal: the dot in the pair's colour, edged so it holds
          // on both woods, as the play board draws it.
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
          // Reached straight and through a portal too: a bead of the pair's colour on the
          // square's corner.
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
  // The glow each pool spills onto the squares round it, over them (not on a thumbnail).
  if (!preview.plain) {
    for (const fold of preview.folds || []) {
      if (cellAbsent(fold.square, view)) continue;
      const [gx, gy] = squareXY(fold.square, cell, view);
      out += foldGlowHtml(gx, gy, cell, fold.pair);
    }
  }
  // Every wild piece's walk, in order, over the squares and the pieces (Q852); not on a
  // thumbnail, which is too small to hold an arrow.
  if (!preview.plain) out += walksHtml(preview, cell, view);
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
  const svg = `<svg viewBox="${box}" class="u-w-100p u-h-auto u-d-block">${
    wood.defs()}${out}</svg>`;
  if (!outside) {
    return `<div class="board-frame ${styleClass(woodFrameStyle(identity))}">${svg}</div>${
      handStripHtml(preview)}`;
  }
  return `<div class="still-named"><div class="board-frame ${
    styleClass(woodFrameStyle(identity))}"><div class="board-box">${svg}<div class="coords${
    view.hex ? ' on-wood' : ''}" aria-hidden="true">${edgeNamesHtml(cell, view, box)}</div></div></div></div>${
    handStripHtml(preview)}`;
}

// The squares' names set outside a board, one `.co` each beside its own edge cell, placed
// as a share of the drawing's `box` (its viewBox) and set off outwards by `--co-out`. The
// play board's `drawCoords` and the designer's bench both draw them here, so the two name
// their squares alike.
function edgeNamesHtml(cell, view, box) {
  const frame = String(box).split(/\s+/).map(Number);
  return edgeLabels(cell, view).map(label => `<span class="co ${styleClass(`left:${
    ((label.x - frame[0]) / frame[2] * 100).toFixed(3)}%; top:${
    ((label.y - frame[1]) / frame[3] * 100).toFixed(3)}%; --dx:${label.dx.toFixed(3)}; --dy:${
    label.dy.toFixed(3)}`)}">${esc(label.text)}</span>`).join('');
}

// One piece of a still, as the board draws it, from its board byte: its icon, turned only
// where the piece has a facing (a single-colour set's 180 is ownership; the play board's
// rule, read off the same list), else its lettered disc. Never an army's icon for a wild
// piece (owner 2, schema 9): an icon pair has one drawing per player, and the third owner
// is neither.
function stillPieceHtml(preview, value, x, y, cell) {
  const stride = preview.sideStride || preview.nTypes;
  const type = (value - 1) % preview.nTypes, player = Math.floor((value - 1) / stride);
  const facing = Math.floor(((value - 1) % stride) / preview.nTypes);
  const icon = player > 1 ? null : ((preview.icons || [])[type] || [])[player];
  if (icon) {
    const heading = (preview.directional || [])[type]
      ? (preview.facingAngles || [])[player] : null;
    const angle = heading ? heading[facing] || 0 : (icon.rotate ? 180 : 0);
    return `<image href="${esc(icon.href)}" x="${x + cell * 0.06}" y="${y + cell * 0.06}"
      width="${cell * 0.88}" height="${cell * 0.88}" preserveAspectRatio="xMidYMid meet"${
      angle ? ` transform="rotate(${angle} ${x + cell / 2} ${y + cell / 2})"` : ''}/>`;
  }
  return `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.34}"
    fill="${player === 0 ? 'var(--p0)' : player === 1 ? 'var(--p1)' : 'var(--p2)'}"
    stroke="var(--ink)" stroke-width="${cell * 0.025}"/>
    <text x="${x + cell / 2}" y="${y + cell / 2}" font-size="${cell * 0.4}"
      text-anchor="middle" dominant-baseline="central" font-weight="600"
      fill="${player === 0 ? 'var(--p0-ink)' : player === 1 ? 'var(--p1-ink)' : 'var(--p2-ink)'}"
      >${esc(preview.pieceIds[type])}</text>`;
}

// A still fold's words, for a screen reader, as the play board's `foldOn` says them.
function stillFoldLabel(preview, end, square, there) {
  const W = preview.width;
  const name = at => squareName(at % W, Math.floor(at / W));
  const side = player => ((preview.sides || [])[player]) || `player ${player}`;
  const where = type => ((preview.folds || []).find(one => one.type === type) || {}).square;
  const from = portalSources(preview.portals, end).map(source => where(source.type))
    .filter(at => at !== undefined).map(name);
  return foldWords(end, name(square), portalPairs(preview.portals), side,
                   there === undefined ? null : name(there), from,
                   portalNamer(preview.pieceIds, preview.pieceNames)).label;
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
// ends ringed, the end picked up to move ringed dashed, the first end of a pair being put
// down, a refused click's square outlined once, and the keyboard's square.
function benchMarksHtml(bench, square, x, y, cell, view, preview) {
  let out = '';
  const cx = x + cell / 2, cy = y + cell / 2;
  for (const pending of bench.pending || []) {
    if (pending.square !== square) continue;
    if (pending.waiting) {
      // An end of a group still being put down (`docs/archive/plan-portal-look.md` §5; a group of
      // two or more since schema 21): a sunk well in the group's colour on a pad of stone,
      // ringed dashed in that colour over the keyline, an end that is not a portal yet.
      const r = cell * 0.46;
      const dash = `stroke-dasharray="${(cell * 0.11).toFixed(3)} ${(cell * 0.07).toFixed(3)}"`;
      out += `<g class="pg-pending" data-pending="${pending.pair}">
        <circle cx="${cx}" cy="${cy}" r="${r}" fill="var(--portal-stone)" opacity="0.55"/>`;
      out += portalGlyph({ pair: pending.pair, end: pending.end, movedBy: [], open: [true, true] },
        x, y, cell, { numbered: true,
          label: `portal group ${pending.pair}: an end waiting for the group to be put down` });
      out += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--portal-edge)"
          stroke-width="${cell * 0.075}" ${dash}/>
        <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--pair-${pending.pair})"
          stroke-width="${cell * 0.045}" ${dash}/></g>`;
      continue;
    }
    // An end the still does not draw yet (a check on its way), drawn as its kind is.
    const said = `portal ${pending.end.length > 1 ? 'pair' : 'group'} ${pending.pair}`;
    out += pending.kind === 'fold'
      ? foldGlyph({ pair: pending.pair, end: pending.end, kind: 'fold', dest: 0, movedBy: [], open: [true, true] },
        x, y, cell, { numbered: true, plain: true, label: said })
      : portalGlyph({ pair: pending.pair, end: pending.end, movedBy: [], open: [true, true] },
        x, y, cell, { numbered: true, label: said });
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
  // The picked group's map (schema 21, Q843b): an arrow from each end to the end it leads
  // to, as the play board's group map draws it.
  if (bench && bench.links && bench.links.length) {
    out += `<g data-link="group" pointer-events="none">${bench.links.map(([a, b, pair]) => {
      const [ax, ay] = centre(a), [bx, by] = centre(b);
      return portalArrowHtml(ax, ay, bx, by, cell, pair, `${a}>${b}`);
    }).join('')}</g>`;
  }
  if (map && map.from >= 0) {
    const cased = `stroke="var(--portal-edge)" stroke-width="${cell * 0.12}"
      stroke-linecap="round" opacity="0.55"`;
    const lit = `stroke="var(--hl-last)" stroke-width="${cell * 0.065}" stroke-linecap="round"`;
    for (const transit of map.transits || []) {
      const both = (from, to, a, b) => leg(from, to, a, b, cased) + leg(from, to, a, b, lit);
      // Schema 24 (§5.3R): through a fold, one leg onto it; the move ends on the end it
      // leads to, which the map's dot already marks.
      if (transit.foldIn !== undefined) {
        out += `<g class="benchLeg" data-exit="${transit.to}" data-entry="${transit.foldIn}" data-fold=""
          pointer-events="none">${both(map.from, transit.foldIn, 0.1, 0.3)}</g>`;
        continue;
      }
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
    // is holding anything in it, and no wild piece is walking (Q852).
    handsUsed: false,
    walks: undefined,
    // ...and no line of sight is looking at it (the effects were the full board's).
    effects: undefined,
  });
}

// ------------------------------------------------------------- a wild piece's walk
//
// Q852 (`docs/reference/designer-case-study-wolves.md` §3 fix 3): each walk the payload
// sends (`viz/walks.py`) as a path of arrows from square to square in the order its pieces
// step, the last back to the first, in the walk's own hue; a step to a square that does not
// touch dashed; the pieces standing on it ringed in that hue, and drawn faintly where the
// board holds none (a draft that is not a game yet draws its walks on a stand-in's board,
// which may not have them). The squares, their order, the starts and which steps jump are
// all the payload's; this only lays them out, and takes a seam the way a wild piece does
// (it owns no seat), as the play board's coming arrows do.

// How many hues walks take in turn (`--walk-1` to `--walk-4`).
const WALK_HUES = 4;
// Where a step leaves a square and stops short of the next, in cells from the middle: clear
// of a piece's disc where one stands there, near the middle where none does, so the arrows
// of an empty stretch read as one path.
const WALK_CLEAR = { piece: 0.5, empty: 0.13 };
// A step drawn by more than one walk, or straight back, is set to its own right by this
// much a time, so two walks over the same squares are both seen.
const WALK_APART = 0.17;

function walkColour(index) {
  return `var(--walk-${(index % WALK_HUES) + 1})`;
}

function walksHtml(preview, cell, view) {
  const walks = preview.walks || [];
  if (!walks.length) return '';
  const W = preview.width, H = preview.height;
  const allow = wrapAllow(preview.wrap, null);
  const stride = preview.sideStride || preview.nTypes;
  const centre = square => {
    const xy = squareXY(square, cell, view);
    return [xy[0] + cell / 2, xy[1] + cell / 2];
  };
  const standing = new Set();
  (preview.board || []).forEach((value, square) => { if (value) standing.add(square); });
  walks.forEach(walk => (walk.starts || []).forEach(square => standing.add(square)));
  const clear = square => cell * (standing.has(square) ? WALK_CLEAR.piece : WALK_CLEAR.empty);
  // Every step drawn, and how many walks draw each pair of squares (either way round).
  const steps = [], shared = new Map();
  walks.forEach((walk, index) => {
    const squares = walk.squares || [];
    if (squares.length < 2) return;
    const jumps = new Set((walk.leaps || []).map(leap => leap.from + '>' + leap.to));
    squares.forEach((from, at) => {
      const to = squares[(at + 1) % squares.length];
      const pair = Math.min(from, to) + '-' + Math.max(from, to);
      shared.set(pair, (shared.get(pair) || 0) + 1);
      steps.push({ index, from, to, pair, leap: jumps.has(from + '>' + to) });
    });
  });
  const seen = new Map();
  let out = '';
  for (const step of steps) {
    const times = seen.get(step.pair) || 0;
    seen.set(step.pair, times + 1);
    const apart = shared.get(step.pair) > 1 ? cell * WALK_APART * (times + 0.5) : 0;
    const colour = walkColour(step.index);
    const attrs = `data-mark="walk" data-walk="${step.index}" data-step="${step.from}>${step.to}"${
      step.leap ? ' data-leap="1"' : ''}`;
    const [ax, ay] = centre(step.from), [bx, by] = centre(step.to);
    // A jump is drawn straight across the board as it is drawn; a step, the way round the
    // seams a wild piece takes, so a step across a seam runs out of one edge and in at the
    // other, each half its own arrow.
    const d = step.leap ? { files: 0, ranks: 0, wrapped: false }
      : wrapStep(step.from, step.to, W, H, allow);
    const [dx, dy] = step.leap ? [bx - ax, by - ay] : stepXY(d, cell, view);
    const length = Math.hypot(dx, dy);
    if (!length) continue;
    const ux = dx / length, uy = dy / length, sx = -uy * apart, sy = ux * apart;
    const piece = (x1, y1, x2, y2) => walkLine(x1 + sx, y1 + sy, x2 + sx, y2 + sy, cell, colour,
                                               step.leap, attrs);
    const from = clear(step.from), to = clear(step.to);
    if (length <= from + to) continue;
    if (d.wrapped) {
      // Out of the edge from the square it leaves, then in from the far edge to the next.
      out += piece(ax + ux * from, ay + uy * from, ax + dx, ay + dy);
      out += piece(bx - dx, by - dy, bx - ux * to, by - uy * to);
    } else {
      out += piece(ax + ux * from, ay + uy * from, ax + dx - ux * to, ay + dy - uy * to);
    }
  }
  // Each piece on a walk ringed in its hue, and drawn faintly where the board has none.
  walks.forEach((walk, index) => {
    // A start is never an absent corner (the validator refuses a route through one).
    for (const square of (walk.starts || []).filter(at => !cellAbsent(at, view))) {
      const [x, y] = squareXY(square, cell, view);
      if (!(preview.board || [])[square] && walk.type !== null && walk.type !== undefined) {
        out += `<g opacity="0.55" data-walkghost="${square}">${
          stillPieceHtml(preview, 2 * stride + walk.type + 1, x, y, cell)}</g>`;
      }
      out += `<circle cx="${(x + cell / 2).toFixed(3)}" cy="${(y + cell / 2).toFixed(3)}"
        r="${(cell * 0.43).toFixed(3)}" fill="none" stroke="var(--walk-edge)"
        stroke-width="${(cell * 0.11).toFixed(3)}"/><circle cx="${(x + cell / 2).toFixed(3)}"
        cy="${(y + cell / 2).toFixed(3)}" r="${(cell * 0.43).toFixed(3)}" fill="none"
        stroke="${walkColour(index)}" stroke-width="${(cell * 0.055).toFixed(3)}"
        data-walkstart="${index}"/>`;
    }
  });
  return `<g data-walks="${walks.length}" pointer-events="none">${out}</g>`;
}

// The line under a still board naming each walk that jumps, in its hue: *Wild pieces 1's
// walk steps from h4 to a5, seven files apart.* The sentence is the payload's
// (`viz.walks`); '' where no walk jumps. `named` (the designer, whose cards and refusals
// name walks *wild pieces 1*, *wild pieces 1's mirror*) first says which hue is which walk,
// where there is more than one, so a refusal naming two walks can be found on the board.
function walkLeapsHtml(preview, named) {
  const walks = (preview && preview.walks) || [];
  const swatch = index => `<span class="walkSwatch ${styleClass(`background:${walkColour(index)}`)}"
    aria-hidden="true"></span>`;
  const names = named && walks.length > 1
    ? `<div data-walknames>${walks.map((walk, index) => `<span class="walkName">${swatch(index)}${
      esc(walk.name)}</span>`).join(' ')}</div>` : '';
  const said = walks.map((walk, index) => walk.says
    ? `<div data-walkleap="${index}">${swatch(index)}${esc(walk.says)}</div>` : '').join('');
  return names || said ? `<div class="small walkLeaps" data-walkleaps>${names}${said}</div>` : '';
}

// Whether a still board draws walks, for its legend line (`MARK_KINDS.walk`).
function walksDrawn(preview) {
  return !!(preview && !preview.plain && (preview.walks || []).some(walk => (walk.squares || []).length > 1));
}
