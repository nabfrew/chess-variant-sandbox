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

function boardPreview(preview, map, drawn) {
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
        if (icon) {
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
        if (moves.has(square)) {
          out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.14}"
            fill="var(--dot)" opacity="0.9"/>`;
        }
        if (takes.has(square)) {
          out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}" r="${cell * 0.38}"
            fill="none" stroke="var(--bad)" stroke-width="${cell * 0.06}" opacity="0.85"/>`;
        }
      }
      out += '</g>';
    }
  }
  // The seam, where the rules put one. A preview is one repeat drawn in its own frame,
  // so the seam falls on the low edge and its copy on the high one -- the same line
  // twice. The frame opens by an arrowhead's reach on that axis: the markers point out
  // of the board and there are no ghost rings here to hold them.
  const seams = seamAxes(preview.wrap);
  const mx = seams.files ? cell * (SEAM_ARROW + 0.05) : 0;
  const my = seams.ranks ? cell * (SEAM_ARROW + 0.05) : 0;
  // A hexagon's sides every player crosses both ways are drawn as repeat lines, since a
  // still has no ghost copy beyond them to say so.
  if (view.cut) out += outlineHtml(cell, view, preview.wrap);
  out += barrierHtml(cell, view, preview.wrap);
  const box = view.hex ? viewBox(view, cell) : `${-mx} ${-my} ${W * cell + 2 * mx} ${H * cell + 2 * my}`;
  return `<div class="board-frame" style="${woodFrameStyle(identity)}"><svg viewBox="${
    box}" style="width:100%; height:auto; display:block">${wood.defs()}${
    out}</svg></div>${handStripHtml(preview)}`;
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
function loneBoard(preview, square, value) {
  return Object.assign({}, preview, {
    board: preview.board.map((_, at) => at === square ? value : 0),
    // One piece on an empty board is a diagram of what a descriptor means, and nobody
    // is holding anything in it.
    handsUsed: false,
  });
}
