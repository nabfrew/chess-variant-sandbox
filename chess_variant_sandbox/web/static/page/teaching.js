// Teaching a move, and where the pieces start.
//
// The board is an input. A person stands a piece on it, clicks a square it should be able
// to reach, and the designer answers with the readings of that click -- one sentence per
// way of moving the rule language can express that gets the piece there. Picking one is
// the move. Nobody writes a vector.
//
// **Every one of those sentences is the server's.** `web/designing.teach` proposes the
// descriptors, asks the engine which squares each one reaches, and says each one back
// through `spec/describe.plain_move_sentence`. This file sends the clicks and draws the
// answer; it works out no square, ranks nothing, and has no opinion about what a
// descriptor means. The page used to hold a fitter of its own, which was a second answer
// to a question the engine was already answering.
//
// **A click means move or take, unless somebody says otherwise.** That is option A10 of
// `docs/plan-designer-ux.md`: drawing where a piece moves and drawing where it takes used
// to be two passes over the board, so a bishop was twenty-six clicks and the second pass
// said the same thing as the first.
//
// Clicking or dragging onto the opening board writes placements, which is the other half
// of this file and unchanged.

// ---------------------------------------------------- teaching a way of moving

let DRAFT_SQUARE = -1,   // where the lone board stands the piece; -1 asks for the middle
    TEACHING = null,     // the move being taught, or null
    PLACING = null;      // what a click on the opening board puts down

// What a click is for. Both first, because a piece takes where it moves unless somebody
// says otherwise -- the two passes this replaces made a bishop twenty-six clicks.
const TEACH_MEANS = [['both', 'move or take'], ['move', 'only to move'],
                     ['capture', 'only to take']];

// A move begins being taught with the piece where the lone board is already standing it.
function startTeaching() {
  const map = DRAFT_MAP;
  if (!map || map.live || map.from < 0) return;
  TEACHING = {
    piece: DRAFT_PIECE, square: map.from, value: map.value, targets: [],
    mean: 'both', fromRank: false, offers: [], chosen: 0, busy: false, error: '',
    asked: 0,
  };
  drawBoardSide();
}

function stopTeaching() {
  TEACHING = null;
  drawBoardSide();
}

// A click on the lone board: a square the piece should reach while a move is being
// taught, and otherwise where to stand the piece.
function teachClick(square) {
  if (offBoard(lastPreview(), square)) return;
  // A portal stays where the setup puts it (N5), and no piece stands on one.
  if (stillPortalAt(lastPreview(), square)) {
    benchSay(`A portal stands on ${boardName(square)}: click another square to stand the piece`
      + ' on, or a square it can come out on to see how it gets there.');
    return;
  }
  if (!TEACHING) {
    DRAFT_SQUARE = square;
    loadMoveMap();
    return;
  }
  if (square === TEACHING.square) return;
  const at = TEACHING.targets.indexOf(square);
  if (at >= 0) TEACHING.targets.splice(at, 1); else TEACHING.targets.push(square);
  askTeach();
}

// What the clicks so far could mean, asked of the server. Every answer is checked
// against the teaching it was asked for, because a second click lands while the first
// is still in the air and an answer about fewer squares would quietly replace it.
function askTeach() {
  const teaching = TEACHING;
  if (!teaching) return;
  const asked = ++teaching.asked;
  teaching.error = '';
  if (!teaching.targets.length) {
    teaching.offers = [];
    teaching.chosen = 0;
    teaching.busy = false;
    drawBoardSide();
    return;
  }
  teaching.busy = true;
  drawBoardSide();
  api('/api/design/teach', {
    spec: DRAFT, piece: teaching.piece, square: teaching.square,
    targets: teaching.targets, mean: teaching.mean, fromRank: teaching.fromRank,
  }).then(answer => {
    if (TEACHING !== teaching || teaching.asked !== asked) return;
    teaching.offers = answer.offers || [];
    teaching.chosen = 0;
    teaching.busy = false;
    drawBoardSide();
  }).catch(error => {
    if (TEACHING !== teaching || teaching.asked !== asked) return;
    teaching.error = errorWords(error);
    teaching.offers = [];
    teaching.busy = false;
    drawBoardSide();
  });
}

function teachingOffer() {
  return (TEACHING && TEACHING.offers[TEACHING.chosen]) || null;
}

// The reading that was picked, written into the piece. The descriptor is the server's,
// word for word: the page never edits one, because a descriptor the page touched is a
// descriptor nothing has asked the engine about.
function keepTeaching(replace) {
  const offer = teachingOffer();
  if (!offer) return;
  const piece = DRAFT.pieces[TEACHING.piece];
  piece.moves = (replace ? [] : piece.moves || []).concat(
    [JSON.parse(JSON.stringify(offer.move))]);
  DRAFT_SQUARE = TEACHING.square;
  TEACHING = null;
  drawDesigner();
  checkDraft();
  loadMoveMap();
}

// ------------------------------------------------------------ the teaching panel

function teachPanelHtml() {
  const preview = lastPreview();
  const piece = (DRAFT.pieces || [])[DRAFT_PIECE] || {};
  const called = esc(piece.name || piece.id || 'this piece');
  if (!preview) return '';
  if (!TEACHING) {
    return `<div class="row small" style="margin-top:8px">
      <button id="dTeach"${DRAFT_MAP && !DRAFT_MAP.live && DRAFT_MAP.from >= 0
        ? '' : ' disabled'}>teach it a move</button>
      <span class="muted">Or click the board to stand the ${called} somewhere else.</span>
    </div>`;
  }
  const W = preview.width;
  const here = squareName(TEACHING.square % W, Math.floor(TEACHING.square / W));
  const rank = Math.floor(TEACHING.square / W) + 1;
  return `<div class="teachPanel">
    <div class="muted small">The ${called} stands on <b>${here}</b>. Click a square it
      should be able to reach, and pick what that click means. Click a square again to
      take it back.</div>
    <div class="row small" style="margin-top:6px">
      ${TEACH_MEANS.map(([value, label]) => `<button data-mean="${value}"
        class="${TEACHING.mean === value ? 'on' : ''}">${esc(label)}</button>`).join('')}
      <label class="pick small" style="align-self:center"><input type="checkbox"
        id="dTeachRank"${TEACHING.fromRank ? ' checked' : ''}> only from rank ${rank}</label>
    </div>
    ${teachOffersHtml(W)}
    <div class="row" style="margin-top:8px">
      <button class="primary" id="dTeachKeep"${teachingOffer()
        ? '' : ' disabled'}>add this to its moves</button>
      <button id="dTeachReplace"${teachingOffer()
        ? '' : ' disabled'}>make it the only one</button>
      <button id="dTeachStop">stop</button>
    </div>
  </div>`;
}

// The readings, as sentences. Each is a button, and picking one draws its squares on the
// board above -- so what a sentence means is on the board before it is kept.
function teachOffersHtml(W) {
  const teaching = TEACHING;
  if (teaching.error) {
    return `<div class="error" style="margin-top:8px">${esc(teaching.error)}</div>`;
  }
  if (!teaching.targets.length) {
    return `<div class="muted small" style="margin-top:8px">Nothing is clicked yet.</div>`;
  }
  if (teaching.busy && !teaching.offers.length) {
    return `<div class="muted small" style="margin-top:8px">working it out&hellip;</div>`;
  }
  if (!teaching.offers.length) {
    return `<div class="muted small" style="margin-top:8px">No way of moving this game
      can say reaches ${esc(squareNames(teaching.targets, W))} from there. Try fewer
      squares, or write it in the move row under the piece &mdash; the editor can
      describe moves an empty board can't show, such as a jump over a piece.</div>`;
  }
  const exact = teaching.offers.filter(offer => offer.exact).length;
  // Said exactly as a piece's moves are said on its game's page, which is where these
  // sentences come from -- a phrase under the piece's name rather than a sentence about
  // it, so the first letter is the only thing this adds.
  const sentence = says => esc(says.charAt(0).toUpperCase() + says.slice(1));
  return `<ul class="teachOffers">${teaching.offers.map((offer, index) => `
    <li><button data-offer="${index}" class="${index === teaching.chosen ? 'on' : ''}">
      ${sentence(offer.says)}${offer.exact
        ? ''
        : ` <span class="muted">&mdash; and ${offer.extra.length} more square${
            offer.extra.length === 1 ? '' : 's'}</span>`}</button></li>`).join('')}</ul>
    <div class="muted small" style="margin-top:4px">${exact
      ? `The first ${exact === 1 ? 'one reaches' : exact + ' reach'} exactly what you
         clicked; the rest reach more, which is often what you meant.`
      : 'None of these reaches exactly what you clicked, so each reaches more.'}
      Each is a move this editor can already write.</div>`;
}

function squareNames(squares, W) {
  return squares.map(s => squareName(s % W, Math.floor(s / W))).join(', ');
}

// ------------------------------------------------------------ where they start

// The tray beside the board: every piece in the draft, one button each, and a row per
// player where the two armies are placed separately. Buttons only -- what the tray is
// for is said under the board by `paletteNoteHtml`, because the tray is a narrow strip
// and a sentence in it would be a column of single words.
function paletteHtml() {
  const setup = DRAFT.setup || {};
  const preview = lastPreview() || {};
  const players = setup.mirror === false ? [0, 1] : [0];
  const sides = preview.sides || ['first', 'second'];
  const rows = players.map(player => `<div class="row small palette">
    ${players.length > 1
      ? `<span class="muted trayWho">${esc(sides[player])}</span>` : ''}
    ${(DRAFT.pieces || []).map((piece, index) => {
      // The portal piece is put down by the portals' row, never from here (D1).
      if (isPortalPiece(index)) return '';
      const icon = ((preview.icons || [])[index] || [])[player];
      const on = PLACING && !PLACING.erase && PLACING.piece === index && PLACING.player === player;
      return `<button data-placing="${index}.${player}" draggable="true" class="${on ? 'on' : ''}"
        title="${esc(piece.name || piece.id || '')}">${icon
          ? `<img src="${esc(icon.href)}" alt=""${icon.rotate ? ' style="transform:rotate(180deg)"' : ''}>`
          : esc(piece.id || '?')}</button>`;
    }).join('')}
  </div>`).join('');
  return rows + `<div class="row small palette">
    <button data-placing="erase" class="${PLACING && PLACING.erase ? 'on' : ''}"
      title="click a square to lift what is standing on it; on a portal, it takes off the whole pair">take off</button>
  </div>${portalRowHtml()}`;
}

// The same strip, for the modes that ask about one piece rather than put pieces down:
// which piece the board is about. Without it the only way to change that is to scroll
// the form beside the board and click a card, which on a phone is several screens away.
function pieceTrayHtml() {
  const preview = lastPreview() || {};
  return `<div class="row small palette">
    ${(DRAFT.pieces || []).map((piece, index) => {
      if (isPortalPiece(index)) return '';
      const icon = ((preview.icons || [])[index] || [])[0];
      return `<button data-pick="${index}" class="${index === DRAFT_PIECE ? 'on' : ''}"
        title="${esc(piece.name || piece.id || '')}">${icon
          ? `<img src="${esc(icon.href)}" alt=""${icon.rotate ? ' style="transform:rotate(180deg)"' : ''}>`
          : esc(piece.id || '?')}</button>`;
    }).join('')}
  </div>`;
}

function wirePieceTray(root) {
  root.querySelectorAll('[data-pick]').forEach(node => {
    node.onclick = () => {
      const index = Number(node.dataset.pick);
      if (index === DRAFT_PIECE) return;
      DRAFT_PIECE = index;
      DRAFT_SQUARE = -1;
      TEACHING = null;
      drawDesigner();
      drawBoardSide();
      loadMoveMap();
    };
  });
}

// What the tray is for, said under the board.
function paletteNoteHtml() {
  return `${freshNoteHtml(freshUnplaced())}<div class="muted small" style="margin-top:6px">${PLACING
    ? 'Click or drag onto the board. Click the same square again to lift what is there.'
    : 'Pick a piece from the tray, then click squares on the board or drag it there.'}
    ${(DRAFT.setup || {}).mirror !== false
      ? 'The second player gets the same army, so every piece you put down is put down'
        + ' for both.'
      : 'Each player is placed separately, and the tray has a row each.'}</div>`;
}

function wirePalette(root) {
  root.querySelectorAll('[data-placing]').forEach(node => {
    const picked = () => {
      if (node.dataset.placing === 'erase') return { erase: true };
      const [piece, player] = node.dataset.placing.split('.').map(Number);
      return { piece, player };
    };
    node.onclick = () => {
      const next = picked();
      PLACING = PLACING && JSON.stringify(PLACING) === JSON.stringify(next) ? null : next;
      // A piece picked from the tray is the board's tool now, so a pair being put down
      // or an end lifted is put away.
      if (PORTAL_TOOL || PORTAL_LIFT) { PORTAL_TOOL = null; PORTAL_LIFT = null; drawBoardSide(); }
      refreshPalette();
    };
    // Not redrawn here: replacing the node being dragged cancels the drag.
    node.ondragstart = event => {
      PLACING = picked();
      if (event.dataTransfer) event.dataTransfer.setData('text/plain', node.dataset.placing);
    };
  });
}

function refreshPalette() {
  document.querySelectorAll('[data-palette]').forEach(box => {
    box.innerHTML = paletteHtml();
    wirePalette(box);
    wirePortalRow(box);
  });
}

// Whether `square` is one of a hexagon's absent corners (schema 16), which is no cell: the
// tray puts nothing there and no click means it. The board does not draw one, so this
// holds a drop, or a click from a board drawn before the draft became a hexagon, to the
// same rule. Which squares are absent is the preview's `absent`, the compiler's answer.
function offBoard(preview, square) {
  return !!preview && (preview.absent || []).includes(square);
}

// Writes the placement that puts PLACING on `square`, or lifts what is there. Under a
// mirror one row serves both players, so a square on the far side is written as its
// reflection; without one, the second player's rows carry `player: 1`. The board that
// comes back is the engine's opening position.
function placeAt(square) {
  const preview = lastPreview();
  if (!preview || !PLACING || offBoard(preview, square)) return;
  const W = preview.width, H = preview.height;
  const setup = DRAFT.setup = DRAFT.setup || {};
  setup.placements = setup.placements || [];
  const mirror = setup.mirror !== false, rotate = setup.orientation === 'rotate';
  const file = square % W, rank = Math.floor(square / W);
  // The setup's reflection, `schema.reflect_square`, applied to a square the spec names.
  // Not a drawing coordinate: where a square is drawn is `squareXY`'s alone.
  const lastFile = W - 1, lastRank = H - 1;
  const spotFor = player => {
    if (!mirror) return { list: player, name: squareName(file, rank) };
    if (player === 0) return { list: 0, name: squareName(file, rank) };
    return { list: 0, name: rotate ? squareName(lastFile - file, lastRank - rank)
                                   : squareName(file, lastRank - rank) };
  };
  const found = [0, 1].map(player => {
    const spot = spotFor(player);
    const row = setup.placements.find(p => (p.player || 0) === spot.list
      && (p.squares || []).includes(spot.name));
    return row ? { player, spot, row } : null;
  }).filter(Boolean);
  const id = PLACING.erase ? null : ((DRAFT.pieces || [])[PLACING.piece] || {}).id;
  const lifted = id != null && found.some(f => f.player === PLACING.player && f.row.piece === id);
  found.forEach(f => { f.row.squares = f.row.squares.filter(name => name !== f.spot.name); });
  setup.placements = setup.placements.filter(p => (p.squares || []).length);
  if (id != null && !lifted) {
    const spot = spotFor(PLACING.player);
    const row = setup.placements.find(p => p.piece === id && (p.player || 0) === spot.list
      && !('facing' in p));
    if (row) row.squares.push(spot.name);
    else setup.placements.push(Object.assign({ piece: id }, spot.list ? { player: 1 } : {},
                                             { squares: [spot.name] }));
  }
  drawDesigner();
  checkDraft();
}

// A click on the opening position while *where it goes* is up: ask about the piece
// standing there. Which piece that is comes from the preview's own board byte rather
// than from anything this page works out, and a square with nothing on it is ignored.
function askWhereItGoes(square) {
  const preview = lastPreview();
  if (!preview) return;
  const value = preview.board[square];
  if (!value) return;
  const type = (value - 1) % preview.nTypes;
  const player = Math.floor((value - 1) / (preview.sideStride || preview.nTypes));
  if (player !== 0) return;
  DRAFT_PIECE = type;
  DRAFT_SQUARE = square;
  drawDesigner();
  loadMoveMap();
}

// Both boards take clicks: the opening board places pieces and answers questions about
// them, the lone piece's board teaches it a move or stands it somewhere else.
function wireBoards(root) {
  const squareOf = event => {
    const node = event.target && event.target.closest && event.target.closest('[data-square]');
    return node ? Number(node.dataset.square) : null;
  };
  root.querySelectorAll('[data-board="alone"] svg').forEach(svg => {
    svg.onclick = event => { const s = squareOf(event); if (s !== null) teachClick(s); };
  });
  root.querySelectorAll('[data-board="start"] svg').forEach(svg => {
    // A square list being picked takes the click before a piece being put down, and
    // *where it goes* asks the engine about whichever piece was clicked rather than
    // changing the position under it. The portals come before the tray's tool: a pair
    // being put down, an end being moved, or an end picked up (D2).
    svg.onclick = event => {
      const s = squareOf(event);
      if (s === null) return;
      BENCH_FOCUS = -1;
      if (PICKING && PICK_LINE) pickLine(s);
      else if (PICKING) pickSquare(s);
      else if (DRAFT_MODE === 'goes') askWhereItGoes(s);
      else if (portalClick(s)) return;
      else if (PLACING) placeAt(s);
    };
    // An end dragged to another square is moved there (D2): down on the end, up on the
    // square. A press and release on one square is the click above.
    svg.onpointerdown = event => {
      const s = squareOf(event);
      BENCH_DRAG = DRAFT_MODE === 'setup' && !PICKING && !PORTAL_TOOL && s !== null
        && !(PLACING && PLACING.erase)
        && portalEndNamed(boardName(s)) ? s : null;
    };
    svg.onpointerup = event => {
      const s = squareOf(event), from = BENCH_DRAG;
      BENCH_DRAG = null;
      if (from === null || s === null || s === from) return;
      PORTAL_LIFT = null;
      portalClick(from);
      if (PORTAL_LIFT) portalClick(s);
    };
    svg.ondragover = event => { if (PLACING || PORTAL_TOOL) event.preventDefault(); };
    svg.ondrop = event => {
      event.preventDefault();
      const s = squareOf(event);
      if (s === null) return;
      if (PORTAL_TOOL) portalClick(s); else placeAt(s);
    };
  });
  // Hovering a square the piece comes out on, or an end it can go in by, draws the
  // journey and says it under the board (D5).
  root.querySelectorAll('figure.benchBoard').forEach(figure => {
    const svg = figure.querySelector('svg');
    if (svg) {
      svg.onpointerover = event => lightLegs(figure, squareOf(event));
      svg.onpointerleave = () => lightLegs(figure, null);
    }
    figure.onkeydown = event => {
      if (benchKey(event.key)) { event.preventDefault(); event.stopPropagation(); }
    };
    if (BENCH_FOCUS >= 0) lightLegs(figure, BENCH_FOCUS);
  });
  const shut = root.querySelector('#dPortalKeyShut');
  if (shut) shut.onclick = () => { PORTAL_KEY_SHUT = true; drawBoardSide(); };
  root.querySelectorAll('[data-pickline]').forEach(node => {
    node.onclick = () => {
      PICK_LINE = PICK_LINE === node.dataset.pickline ? '' : node.dataset.pickline;
      drawBoardSide();
    };
  });
  const byId = id => root.querySelector('#' + id);
  if (byId('dTeach')) byId('dTeach').onclick = startTeaching;
  if (!TEACHING) return;
  root.querySelectorAll('[data-mean]').forEach(node => {
    node.onclick = () => { TEACHING.mean = node.dataset.mean; askTeach(); };
  });
  root.querySelectorAll('[data-offer]').forEach(node => {
    // Picking a reading only draws it. Nothing reaches the draft until it is kept, so
    // every sentence can be tried on the board and none of them costs an undo.
    node.onclick = () => { TEACHING.chosen = Number(node.dataset.offer); drawBoardSide(); };
  });
  byId('dTeachRank').onchange = event => {
    TEACHING.fromRank = event.target.checked;
    askTeach();
  };
  byId('dTeachKeep').onclick = () => keepTeaching(false);
  byId('dTeachReplace').onclick = () => keepTeaching(true);
  byId('dTeachStop').onclick = stopTeaching;
}
