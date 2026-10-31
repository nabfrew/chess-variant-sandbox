// Squares nothing may occupy, drawn as a wall rather than left looking ordinary.
const WALLS = new Set(DATA.obstacles || []);
// Squares the rules single out, a list per square: the ones the game ends on when a
// piece reaches one, and the ones a piece promotes on. Empty for a ruleset whose rules
// single out no square. Rebuilt when the reader puts the promotion zone away, which is
// the one thing on the board that is theirs to decide.
let ZONES = zonesShown();
let SQUARE_MARKS = squareMarks(DATA, { zone: !ZONES });
const CELL = 100 / Math.max(DATA.width, DATA.height);
const W = DATA.width, H = DATA.height;
const AXES = wrapAxes(DATA.wrap);
// A hexagon's absent corners and its copies (schema 16); `null` on any other board.
const CUT = boardCut(DATA.absent, DATA.wrap);
const STILL = window.matchMedia('(prefers-reduced-motion: reduce)');
// Every position the replay can stand on, in order: the opening, then for each ply the
// mover's move and, where the rules ran a wild-piece phase after it, that phase. The
// split is `plyBeats`, the play board's own, so the two surfaces step through the same
// positions. A game without wild pieces has one beat per ply, which is what it had.
const BEATS = [{ kind: 'start', ply: 0, board: DATA.boards[0], steps: null, walkers: null }];
DATA.moves.forEach((move, index) => {
  for (const beat of plyBeats(DATA.boards[index + 1], move)) {
    BEATS.push(Object.assign({ ply: index + 1 }, beat));
  }
});
let at = 0, timer = null;
// Schema 22: where the folds lie in the position on screen (`DATA.folds`, one list a ply).
let FOLDS_SHOWN = [];
// Which repeat of a wrapping board is on screen. Zero is the frame the squares are
// named in, and nothing else -- see the shared module above.
let SHIFT = { files: 0, ranks: 0 };
let FRAME = boardView(W, H, AXES, SHIFT, false, DATA.lattice, CUT);

const svg = document.getElementById('board');

function pieceAt(board, square) { return pieceOf(board[square]); }

function pieceOf(value) {
  if (!value) return null;
  const index = value - 1;
  // `sideStride` is a whole army's worth of values, not one type's: with a facing a
  // piece has one value per orientation, and dividing by the type count gives the
  // frame rather than the owner.
  return {
    player: Math.floor(index / DATA.sideStride),
    type: index % DATA.nTypes,
    facing: Math.floor((index % DATA.sideStride) / DATA.nTypes),
  };
}

// An icon is drawn where the library has one for the piece's name, and the spec
// letter where it does not -- which is most invented pieces, and some real ones.
function iconFor(piece) {
  // An icon pair is one drawing per player and has no third entry, so a wild piece
  // (owner 2, schema 9) wears the first seat's (PI3, as the play board does); with no
  // drawing it is the lettered disc.
  const pair = (DATA.pieceIcons || [])[piece.type];
  return pair ? pair[piece.player > 1 ? 0 : piece.player] : null;
}

// PI3, as `stillboard.js` draws it: a wild piece's drawing is a size smaller on the third
// owner's disc, edged in `--ink` (not `--p2`, the trail's stroke); the duck stays bare.
const WILD_ICON = 0.74;

// The disc a lettered piece is drawn on. `--p2` is the wild third owner's.
function tintFor(player) {
  return player === 0 ? ['var(--p0)', 'var(--p0-ink)']
       : player === 1 ? ['var(--p1)', 'var(--p1-ink)']
       : ['var(--p2)', 'var(--p2-ink)'];
}

function drawPiece(piece, x, y, idle) {
  const cx = x + CELL / 2, cy = y + CELL / 2;
  // Schema 19: a portal end, drawn as the play board draws it (Q838b's vortex), its arms
  // turning idly on its own square and still on a ghost copy or a piece being moved.
  // Its beak (schema 21) points at the end it leads to on the board on screen, the nearest
  // copy round a wrapping board.
  const end = DATA.portals ? DATA.portals.ends.find(one => one.type === piece.type) : null;
  if (end) {
    const to = portalPartner(DATA.portals, end);
    const shown = BEATS && BEATS[at] ? BEATS[at].board : null;
    const there = to && shown ? portalSquareIn(shown, DATA.nTypes, to.type) : -1;
    return portalGlyph(end, x, y, CELL, { numbered: portalPairs(DATA.portals) > 1,
                                          idle: !!idle, flipped: FRAME.flipped,
                                          aim: portalAimFor(end, there, x, y, CELL, FRAME) });
  }
  const icon = iconFor(piece);
  if (icon) {
    const based = piece.player > 1 && DATA.duck !== piece.type;
    const size = CELL * (based ? WILD_ICON : 0.88), corner = (CELL - size) / 2;
    const disc = based
      ? `<circle cx="${cx}" cy="${cy}" r="${CELL * 0.44}" fill="var(--p2)" fill-opacity="0.6"
          stroke="var(--ink)" stroke-width="${CELL * 0.025}" data-wild="1"/>` : '';
    // A piece with a facing is drawn pointing where it faces, and nothing else turns.
    // `directional` names the types that have one; `facingAngles` gives the heading,
    // that table rather than `facing * 90`, because a mirrored army turns the other
    // way round on screen. The one other rotation is ownership: a single-colour set
    // (shogi) has both armies in one colour and marks the second by turning the tile.
    const heading = (DATA.directional || [])[piece.type]
      ? (DATA.facingAngles || [])[piece.player] : null;
    const angle = heading ? heading[piece.facing] || 0 : (icon.rotate ? 180 : 0);
    const spin = angle ? ` transform="rotate(${angle} ${cx} ${cy})"` : '';
    return `${disc}<image href="${icon.href}" x="${x + corner}" y="${y + corner}"
             width="${size}" height="${size}"
             preserveAspectRatio="xMidYMid meet"${spin}/>`;
  }
  const [fill, ink] = tintFor(piece.player);
  return `<circle cx="${cx}" cy="${cy}" r="${CELL * 0.36}"
           fill="${fill}" stroke="var(--ink)" stroke-width="${CELL * 0.025}" opacity="0.95"/>`
       + `<text x="${cx}" y="${cy}" fill="${ink}"
           font-size="${CELL * 0.42}" font-weight="600" text-anchor="middle"
           dominant-baseline="central">${DATA.pieceIds[piece.type]}</text>`;
}

// One square, wherever in the grid it is being drawn. A ghost copy of it -- the board
// continuing past the outline -- draws the same things at the ring's own paleness, and
// takes neither the coordinate label, which belongs to the frame the outline marks, nor
// the `data-piece` tag, which has to name one node for the crossing to find.
//
// `marked` is the squares the beat on screen changed: the mover's two, or the two of
// every wild piece that walked in a phase -- which is where a reader looks to see what
// just happened, and for a wolf is the only way to see where it came from.
function cellHtml(board, marked, square, u, v, depth) {
  const place = cellXY(u, v, CELL, FRAME);
  const x = place[0], y = place[1];
  const file = square % W, rank = Math.floor(square / W);
  let out = cellShape(x, y, CELL, FRAME, `fill="${cellShade(file, rank, FRAME)}"`);
  if (WALLS.has(square)) {
    out += cellShape(x, y, CELL, FRAME, 'fill="var(--wall)"');
    out += `<path d="${wallCross(x, y, CELL, FRAME)}"
             stroke="var(--ink)" stroke-width="${CELL * 0.06}" opacity="0.5"/>`;
  }
  if (marked.has(square)) {
    out += cellShape(x, y, CELL, FRAME, 'fill="var(--mark)" opacity="0.42"');
  }
  // Under the piece, because a square that ends the game is a square pieces arrive on.
  if (SQUARE_MARKS[square]) out += squareBadge(x, y, CELL, SQUARE_MARKS[square], FRAME, 'replay');
  // Schema 22 (Q843d): a fold, under any piece standing in it, as the play board draws it.
  const fold = (FOLDS_SHOWN || []).find(one => one.square === square);
  const foldEnd = fold && DATA.portals ? DATA.portals.ends.find(one => one.type === fold.type) : null;
  if (foldEnd) {
    const to = portalPartner(DATA.portals, foldEnd);
    const there = to ? ((FOLDS_SHOWN || []).find(one => one.type === to.type) || { square: -1 }).square : -1;
    out += foldGlyph(foldEnd, x, y, CELL, { numbered: portalPairs(DATA.portals) > 1,
      occupied: !!pieceAt(board, square), idle: depth === 0, plain: depth !== 0,
      flipped: FRAME.flipped, aim: portalAimFor(foldEnd, there, x, y, CELL, FRAME) });
  }
  const piece = pieceAt(board, square);
  if (piece) {
    out += depth === 0
      ? `<g data-piece="${square}">${drawPiece(piece, x, y, true)}</g>`
      : drawPiece(piece, x, y);
  }
  if (depth === 0) return out + coordsHtml(square, x, y, CELL, FRAME);
  return `<g opacity="${ghostFade(depth).toFixed(3)}">${out}</g>`;
}

function draw() {
  const beat = BEATS[at];
  const board = beat.board;
  const move = beat.ply > 0 ? DATA.moves[beat.ply - 1] : null;
  const marked = new Set(beat.kind === 'phase' ? beat.steps.flatMap(s => [s.from, s.to])
                         : move ? [move.from, move.to] : []);
  // A move through a portal marks the two ends it passed too (schema 19), and a move
  // through a fold the fold and the end it leads to (schema 22).
  if (beat.kind !== 'phase' && move && move.portal !== undefined) {
    marked.add(move.portal);
    marked.add(move.portalOut);
  }
  if (beat.kind !== 'phase' && move && move.foldIn !== undefined) {
    marked.add(move.foldIn);
    marked.add(move.foldOut);
  }
  // Q1074bo: a push marks every square its line moved onto, and the square of the piece it
  // threw off or crushed, since that piece is gone and nothing else says where it stood.
  if (beat.kind !== 'phase' && move && Array.isArray(move.pushed)) {
    for (const [square, onto] of move.pushed) marked.add(onto === null ? square : onto);
  }
  FOLDS_SHOWN = DATA.folds ? DATA.folds[beat.ply] || [] : [];
  FRAME = boardView(W, H, AXES, SHIFT, false, DATA.lattice, CUT);
  svg.setAttribute('viewBox', viewBox(FRAME, CELL));

  // The board continuing past the outline first, then the repeat being read, then the
  // outline over both. A board with edges has no ghosts and no outline.
  let out = '';
  for (const ghost of ghostCells(FRAME)) {
    out += cellHtml(board, marked, ghost.square, ghost.u, ghost.v, ghost.depth);
  }
  for (let rank = H - 1; rank >= 0; rank--) {
    for (let file = 0; file < W; file++) {
      const square = rank * W + file;
      if (cellAbsent(square, FRAME)) continue;
      const place = squareCell(square, FRAME);
      out += cellHtml(board, marked, square, place[0], place[1], 0);
    }
  }
  // The outline, then the seam over it: the outline stands down on an axis whose seam
  // is a fact, so the two never draw two boundaries on one axis.
  svg.innerHTML = markDefs(SQUARE_MARKS, 'replay') + out + outlineHtml(CELL, FRAME, DATA.wrap)
                      + barrierHtml(CELL, FRAME, DATA.wrap);
  drawShift();
  drawMarks();
  drawPurses(beat.ply);

  document.getElementById('ply').textContent = `${beat.ply} / ${DATA.moves.length}`;
  document.getElementById('material').textContent =
    move && move.material !== null ? move.material.toFixed(2) : '-';
  document.getElementById('branching').textContent =
    move && move.branching !== null ? move.branching : '-';

  document.querySelectorAll('.move').forEach(node => {
    node.classList.toggle('active', Number(node.dataset.beat) === at);
  });
  const last = BEATS.length - 1;
  document.getElementById('start').disabled = at === 0;
  document.getElementById('prev').disabled = at === 0;
  document.getElementById('next').disabled = at >= last;
  document.getElementById('end').disabled = at >= last;
}

// TC7 (Q1049): each seat's purse in the position on screen, worded as the play board words
// it (`purseWords`, `purseTitle`). The counts are the core's: `DATA.purses` holds one pair
// per ply, read off the replayed state, so a step back shows that position's coins. A phase
// beat stands on its ply's position, so it shows the purses the ply left.
function drawPurses(ply) {
  if (!DATA.purse || !DATA.purses) return;
  const purse = DATA.purses[ply] || [0, 0];
  document.querySelectorAll('[data-purse]').forEach(node => {
    node.textContent = purseWords(purse[Number(node.dataset.purse)] || 0);
  });
}

function seatsHtml() {
  const sides = DATA.sides || ['white', 'black'];
  const title = esc(purseTitle(DATA.purse)).replace(/"/g, '&quot;');
  return [0, 1].map(player => `<div><b>${esc(sides[player])}</b> <span class="muted">${
    esc(DATA.meta.agents[player])}</span><div data-purse="${player}" title="${title}"></div></div>`)
    .join('');
}

// Moving the view across the seam, one square at a time. A wrapping board has no
// boundary of its own, so this says which repeat is on screen; `reset` goes back to the
// frame the squares take their names from. A board with edges gets no row at all.
function drawShift() {
  const row = document.getElementById('shift');
  if (!AXES.files && !AXES.ranks) { row.innerHTML = ''; return; }
  if (row.dataset.built !== 'yes') {
    const across = AXES.files ? `
      <button data-shift="-1,0" title="one file back (shift + left arrow)">&larr;</button>
      <button data-shift="1,0" title="one file on (shift + right arrow)">&rarr;</button>` : '';
    const up = AXES.ranks ? `
      <button data-shift="0,1" title="one rank up (shift + up arrow)">&uarr;</button>
      <button data-shift="0,-1" title="one rank down (shift + down arrow)">&darr;</button>` : '';
    row.innerHTML = `<span class="muted">shift the view</span>${across}${up}
      <button id="reframe" title="back to the frame the squares are named in">reset</button>
      <span class="muted" id="frame"></span>`;
    row.dataset.built = 'yes';
    row.querySelectorAll('[data-shift]').forEach(node => {
      const by = node.dataset.shift.split(',');
      node.onclick = () => shiftView(Number(by[0]), Number(by[1]));
    });
    document.getElementById('reframe').onclick = () => { SHIFT = { files: 0, ranks: 0 }; draw(); };
  }
  // Which square the repeat on screen runs from, read off the frame rather than off
  // `SHIFT` -- eight clicks along an eight-file board is back where it started.
  const corner = cellSquare(0, 0, FRAME);
  document.getElementById('reframe').disabled = corner === 0;
  document.getElementById('frame').textContent =
    'one repeat, from ' + squareName(corner % W, Math.floor(corner / W));
}

function shiftView(files, ranks) {
  SHIFT = { files: SHIFT.files + files, ranks: SHIFT.ranks + ranks };
  draw();
}

// The promotion zone's toggle, beside the row that shifts the view. A game with no
// promotion gets no button, which is how the control says what it is about.
function drawMarks() {
  const row = document.getElementById('marks');
  if (!DATA.zones) { row.innerHTML = ''; row.dataset.built = ''; return; }
  const state = ZONES ? 'on' : 'off';
  if (row.dataset.built === state) return;
  row.innerHTML = zoneToggleHtml(ZONES);
  row.dataset.built = state;
  document.getElementById('zonetoggle').onclick = toggleZones;
}

function toggleZones() {
  ZONES = !ZONES;
  rememberZones(ZONES);
  SQUARE_MARKS = squareMarks(DATA, { zone: !ZONES });
  draw();
}

// How long a crossing takes to cross. The replay redraws rather than animating, which
// is right for a move you can read off the two marks it leaves -- and wrong for a move
// through the seam, whose two squares sit at opposite ends of the board with nothing
// between them to explain the jump. So this is the one motion the replay has.
const CROSS_MS = 320;

function crossTravel(node, dx, dy) {
  node.animate([{ transform: `translate(${dx}px, ${dy}px)` },
                { transform: 'translate(0px, 0px)' }],
               { duration: CROSS_MS, easing: 'cubic-bezier(.3,.7,.3,1)' });
}

// The two copies a move across the seam is drawn as: the piece arriving, flown in from
// outside the far edge, and a copy of what left, flown out through the near one. One
// translation, one repeat apart, cross-fading in the middle so there is one piece to
// follow rather than two.
//
// A wild piece is nobody's, so `wrapAllow` gives it the union of the seams -- the same
// answer the play board gives it.
function showCrossing(origin, target, before, after) {
  if (STILL.matches) return;
  const arrived = pieceAt(after, target);
  const left = pieceAt(before, origin);
  if (!arrived || !left) return;
  const step = wrapStep(origin, target, W, H, wrapAllow(DATA.wrap, arrived.player));
  if (!step.wrapped) return;
  const node = svg.querySelector(`[data-piece="${target}"]`);
  if (!node) return;
  const d = stepXY(step, CELL, FRAME);
  const from = squareXY(origin, CELL, FRAME);
  svg.appendChild(node);
  crossTravel(node, -d[0], -d[1]);
  node.animate(crossOpacity(false), { duration: CROSS_MS, fill: 'backwards' });
  const away = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  away.innerHTML = drawPiece(left, from[0] + d[0], from[1] + d[1]);
  svg.appendChild(away);
  crossTravel(away, -d[0], -d[1]);
  const fade = away.animate(crossOpacity(true), { duration: CROSS_MS, fill: 'both' });
  fade.onfinish = () => away.remove();
}

// A push (Q1074bo, the play board's since Q1074b5): two positions cannot say which pieces a
// push moved, so this reads the engine's line (`move.pushed`, `viz.wrapped.push_transit`) and
// works out nothing. The pusher travels onto the square it pushed from, each piece of the line
// slides one step on (`move.pushStep`, which is the right way round a seam too), and the
// last, where it left the board, falls away over the edge along the step, or, crushed on a
// wall (`move.pushCrush`), is pressed flat against that side of its square and fades. The
// motion is the play board's `pushFall` and `pushCrush`, in the replay's own few lines,
// since the replay inlines `wrapped.js` and not the page's `motion.js`.
function pushGhost(piece, square) {
  const place = squareXY(square, CELL, FRAME);
  const ghost = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  ghost.innerHTML = drawPiece(piece, place[0], place[1]);
  ghost.style.transformOrigin = `${place[0] + CELL / 2}px ${place[1] + CELL / 2}px`;
  svg.appendChild(ghost);
  return ghost;
}

function showPush(move, before) {
  if (STILL.matches || !Array.isArray(move.pushed) || !Array.isArray(move.pushStep)) return;
  const d = stepXY({ files: move.pushStep[0], ranks: move.pushStep[1] }, CELL, FRAME);
  const step = wrapStep(move.from, move.to, W, H,
    wrapAllow(DATA.wrap, (pieceAt(before, move.from) || {}).player));
  const pusher = svg.querySelector(`[data-piece="${move.to}"]`);
  if (pusher && !step.wrapped) {
    const from = squareXY(move.from, CELL, FRAME), to = squareXY(move.to, CELL, FRAME);
    svg.appendChild(pusher);
    crossTravel(pusher, from[0] - to[0], from[1] - to[1]);
  }
  for (const [square, onto] of move.pushed) {
    if (onto !== null) {
      const node = svg.querySelector(`[data-piece="${onto}"]`);
      if (node) crossTravel(node, -d[0], -d[1]);
      continue;
    }
    const piece = pieceAt(before, square);
    if (!piece) continue;
    const ghost = pushGhost(piece, square);
    ghost.setAttribute('data-journey', move.pushCrush ? 'push-crush' : 'push-fall');
    const reach = Math.hypot(d[0], d[1]) || 1;
    const nx = d[0] / reach * CELL * 0.34, ny = d[1] / reach * CELL * 0.34;
    const flat = Math.abs(d[0]) >= Math.abs(d[1]) ? 'scale(0.18, 1.12)' : 'scale(1.12, 0.18)';
    const frames = move.pushCrush ? [
      { opacity: 1, transform: 'translate(0px, 0px) scale(1, 1)' },
      { opacity: 1, transform: `translate(${nx * 0.5}px, ${ny * 0.5}px) scale(1, 1)`, offset: 0.3 },
      { opacity: 0.85, transform: `translate(${nx}px, ${ny}px) ${flat}`, offset: 0.62 },
      { opacity: 0, transform: `translate(${nx}px, ${ny}px) ${flat}` },
    ] : [
      { opacity: 1, transform: 'translate(0px, 0px) scale(1)' },
      { opacity: 0.9, transform: `translate(${d[0] * 0.7}px, ${d[1] * 0.7}px) scale(0.92)`, offset: 0.45 },
      { opacity: 0, transform: `translate(${d[0] * 1.1}px, ${d[1] * 1.1 + CELL * 0.35}px) scale(0.35)` },
    ];
    const away = ghost.animate(frames, { duration: PUSH_MS, easing: 'ease-in', fill: 'both' });
    away.onfinish = () => ghost.remove();
  }
}

// How long a piece thrown off or crushed takes to go: the play board's `PACE.pushFall`.
const PUSH_MS = 520;

function goto(target) {
  const was = at;
  at = Math.max(0, Math.min(BEATS.length - 1, target));
  draw();
  // Only a step onto the next beat is something happening. Jumping several, or back, is
  // a position being looked up, and there is no one motion to show for it. What moved in
  // a beat is the mover, or every wild piece that walked in a phase.
  if (at !== was + 1) return;
  const beat = BEATS[at];
  const moved = beat.kind === 'phase' ? beat.steps : [DATA.moves[beat.ply - 1]];
  for (const m of moved) showCrossing(m.from, m.to, BEATS[was].board, beat.board);
  if (beat.kind !== 'phase') showPush(DATA.moves[beat.ply - 1], BEATS[was].board);
}

function togglePlay() {
  const button = document.getElementById('play');
  if (timer) {
    clearInterval(timer); timer = null; button.innerHTML = '&#9654; play';
    return;
  }
  if (at >= BEATS.length - 1) at = 0;
  button.innerHTML = '&#10073;&#10073; pause';
  timer = setInterval(() => {
    if (at >= BEATS.length - 1) { togglePlay(); return; }
    goto(at + 1);
  }, 420);
}

const meta = DATA.meta;
document.title = `${meta.ruleset} replay`;
document.getElementById('title').textContent = `${meta.ruleset} [${meta.hash}]`;
const verdict = meta.winner === null
  ? `draw by ${meta.reason}`
  : `player ${meta.winner} wins by ${meta.reason}`;
document.getElementById('subtitle').textContent =
  `${meta.agents[0]} vs ${meta.agents[1]} - seed ${meta.seed} - ${verdict} in ${meta.plies} plies`;

// One row per beat, so a phase is a line of its own under the move it answered, worded
// by `phaseWords` exactly as the play board's move list words it.
const esc = text => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const nameOf = value => {
  const piece = pieceOf(value);
  return piece ? DATA.pieceNames[piece.type] || DATA.pieceIds[piece.type] : null;
};
// Generation 40 (N3c): the compiled pounce kind of a wild piece's type, for its words.
const pounceOf = value => {
  const piece = pieceOf(value);
  return piece ? (DATA.pounceKinds || [])[piece.type] || null : null;
};
document.getElementById('moves').innerHTML = BEATS
  .map((beat, index) => {
    if (beat.kind === 'start') return '';
    const i = beat.ply - 1;
    const said = beat.kind === 'phase'
      ? esc(phaseWords(beat.steps, beat.walkers, nameOf, W, pounceOf))
      : `${i % 2 === 0 ? `${Math.floor(i / 2) + 1}.` : '&nbsp;&nbsp;&hellip;'} ${
        esc(DATA.moves[i].text)}`;
    return `<div class="move${beat.kind === 'phase' ? ' phase' : ''}" data-beat="${index}">${
      said}</div>`;
  })
  .join('');
document.getElementById('moves').addEventListener('click', event => {
  const node = event.target.closest('.move');
  if (node) goto(Number(node.dataset.beat));
});

if (DATA.purse && DATA.purses) {
  const seats = document.getElementById('seats');
  seats.innerHTML = seatsHtml();
  seats.hidden = false;
}

const credits = DATA.credits || [];
document.getElementById('credits').innerHTML = credits.length
  ? '<div>' + credits.map(line => line.replace(/&/g, '&amp;').replace(/</g, '&lt;')).join('</div><div>') + '</div>'
  : '';

document.getElementById('start').onclick = () => goto(0);
document.getElementById('prev').onclick = () => goto(at - 1);
document.getElementById('next').onclick = () => goto(at + 1);
document.getElementById('end').onclick = () => goto(BEATS.length - 1);
document.getElementById('play').onclick = togglePlay;
// The bare arrows step through the game, so shift and an arrow moves the view across
// the seam. Reading a game is the more common of the two, and it keeps the key.
document.addEventListener('keydown', event => {
  if (event.shiftKey && event.key.indexOf('Arrow') === 0) {
    const by = { ArrowLeft: [-1, 0], ArrowRight: [1, 0],
                 ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
    if (!by || (by[0] && !AXES.files) || (by[1] && !AXES.ranks)) return;
    event.preventDefault();
    shiftView(by[0], by[1]);
    return;
  }
  if (event.key === 'ArrowLeft') goto(at - 1);
  if (event.key === 'ArrowRight') goto(at + 1);
  if (event.key === ' ') { event.preventDefault(); togglePlay(); }
});

draw();
