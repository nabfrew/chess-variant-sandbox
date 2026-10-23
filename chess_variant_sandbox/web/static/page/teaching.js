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
    return `<div class="row small u-mt-8">
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
    <div class="row small u-mt-6">
      ${TEACH_MEANS.map(([value, label]) => `<button data-mean="${value}"
        class="${TEACHING.mean === value ? 'on' : ''}">${esc(label)}</button>`).join('')}
      <label class="pick small u-as-center"><input type="checkbox"
        id="dTeachRank"${TEACHING.fromRank ? ' checked' : ''}> only from rank ${rank}</label>
    </div>
    ${teachOffersHtml(W)}
    <div class="row u-mt-8">
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
    return `<div class="error u-mt-8">${esc(teaching.error)}</div>`;
  }
  if (!teaching.targets.length) {
    return `<div class="muted small u-mt-8">Nothing is clicked yet.</div>`;
  }
  if (teaching.busy && !teaching.offers.length) {
    return `<div class="muted small u-mt-8">working it out&hellip;</div>`;
  }
  if (!teaching.offers.length) {
    return `<div class="muted small u-mt-8">No way of moving this game
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
    <div class="muted small u-mt-4">${exact
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
      const icon = ((preview.icons || [])[index] || [])[player];
      const on = PLACING && !PLACING.erase && PLACING.piece === index && PLACING.player === player;
      return `<button data-placing="${index}.${player}" draggable="true" class="${on ? 'on' : ''}"
        title="${esc(trayTitle(piece))}">${icon
          ? `<img src="${esc(icon.href)}" alt=""${icon.rotate ? ' class="u-tf-rot180"' : ''}>`
          : esc(piece.id || '?')}${traySpeedHtml(piece)}</button>`;
    }).join('')}
  </div>`).join('');
  return rows + `<div class="row small palette">
    <button data-placing="erase" class="${PLACING && PLACING.erase ? 'on' : ''}"
      title="click a square to lift what is standing on it; on a portal, it takes off the whole pair">take off</button>
  </div>${portalRowHtml()}${wildRowHtml()}`;
}

// A tray button's name, and its speed where one is set (Q995c, fix 5): in a game in rounds
// the speed decides which of two moves is made first, and it showed only in a fold.
function trayTitle(piece) {
  const speed = pieceSpeedWords(piece);
  return (piece.name || piece.id || '') + (speed ? `, ${speed}` : '');
}

function traySpeedHtml(piece) {
  const speed = pieceSpeedWords(piece);
  return speed ? `<span class="traySpeed" aria-hidden="true">${esc(String(piece.speed))}</span>`
    : '';
}

// The same strip, for the modes that ask about one piece rather than put pieces down:
// which piece the board is about. Without it the only way to change that is to scroll
// the form beside the board and click a card, which on a phone is several screens away.
function pieceTrayHtml() {
  const preview = lastPreview() || {};
  return `<div class="row small palette">
    ${(DRAFT.pieces || []).map((piece, index) => {
      const icon = ((preview.icons || [])[index] || [])[0];
      return `<button data-pick="${index}" class="${index === DRAFT_PIECE ? 'on' : ''}"
        title="${esc(trayTitle(piece))}">${icon
          ? `<img src="${esc(icon.href)}" alt=""${icon.rotate ? ' class="u-tf-rot180"' : ''}>`
          : esc(piece.id || '?')}${traySpeedHtml(piece)}</button>`;
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

// What the tray is for, said under the board. Its own element, so a piece picked in the
// tray is said at once without redrawing the board.
function paletteNoteHtml() {
  return `<div data-palettenote>${paletteNoteBodyHtml()}</div>`;
}

function paletteNoteBodyHtml() {
  return `${freshNoteHtml(freshUnplaced())}<div class="muted small u-mt-6">${PLACING
    ? 'Click or drag onto the board. Click the same square again to lift what is there.'
    : 'Pick a piece from the tray, then click squares on the board or drag it there.'}
    ${(DRAFT.setup || {}).mirror !== false
      ? 'The second player gets the same army, so every piece you put down is put down'
        + ' for both.'
      : 'Each player is placed separately, and the tray has a row each.'}</div>${
    wildTrayNoteHtml()}`;
}

// ------------------------------------------------------------ a wild piece's type

// A type the draft's wild pieces are made of is also in the tray, and the tray's one is a
// player's (the wolves walk, steps 10-12, docs/reference/designer-case-study-wolves.md):
// so picking it, and asking where it goes, say which of the two it is and where the
// other is written. Read off the draft's own `rules.npcs`, entry by entry; no square is
// worked out here, only the ones the entry names.

// The piece ids the draft's wild pieces are, in the order their entries come.
function wildTypes() {
  const npcs = (DRAFT.rules || {}).npcs;
  if (!Array.isArray(npcs)) return [];
  const out = [];
  npcs.forEach(npc => {
    if (npc && typeof npc.piece === 'string' && !out.includes(npc.piece)) out.push(npc.piece);
  });
  return out;
}

// A walk as it is written, as HTML, shortened past six squares to its first three, its
// last and its length.
function walkWords(route) {
  const squares = (route || []).filter(square => typeof square === 'string').map(esc);
  if (squares.length <= 6) return squares.join(', ');
  return `${squares.slice(0, 3).join(', ')} &hellip; ${squares[squares.length - 1]}
    (${squares.length} squares)`;
}

// The wild pieces of type `id`, one phrase per entry: where it starts, the others on its
// walk, its mirror, and the walk itself.
function wildOnesWords(id) {
  const npcs = (DRAFT.rules || {}).npcs;
  if (!Array.isArray(npcs)) return [];
  return npcs.filter(npc => npc && npc.piece === id && Array.isArray(npc.route)
                     && npc.route.length).map(npc => {
    const more = Array.isArray(npc.also_at) ? npc.also_at.filter(s => typeof s === 'string') : [];
    const where = [npc.route[0]].concat(more).map(esc).join(' and ');
    const stands = more.length ? `the ones on ${where}` : `the one on ${where}`;
    const walks = more.length ? 'walk' : 'walks';
    const mirrored = npc.mirror === true ? ', with a mirrored one,' : '';
    if (npc.route.length === 1) return `${stands}${mirrored} never moves`;
    return `${stands}${mirrored} ${walks} ${walkWords(npc.route)}`;
  });
}

function wildCardLinkHtml() {
  return '<a href="#" data-wildcard>wild pieces</a>';
}

// The tray's piece, when its type is a wild piece's too: one line, under the tray's own.
function wildTrayNoteHtml() {
  if (!PLACING || PLACING.erase) return '';
  const piece = (DRAFT.pieces || [])[PLACING.piece];
  if (!piece || !wildTypes().includes(piece.id)) return '';
  const called = esc(piece.name || piece.id);
  const sides = (lastPreview() || {}).sides || ['first', 'second'];
  const whose = (DRAFT.setup || {}).mirror !== false ? 'one for each player'
    : `for the ${esc(sides[PLACING.player] || 'first')} player`;
  return `<div class="small wildNote u-mt-6" data-wildnote>The
    <b>${called}</b> is also a wild piece in this game. The tray puts down a player's ${called}, ${whose};
    a wild ${called} and its walk are on the ${wildCardLinkHtml()} card.</div>`;
}

// The *wild pieces* link: the rule's card, scrolled to. It is a long way down the form.
function wireWildLinks(root) {
  (root || document).querySelectorAll('[data-wildcard]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const card = document.querySelector('.ruleCard[data-rule="npcs"]');
      if (card && typeof card.scrollIntoView === 'function') {
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    };
  });
}

// ------------------------------------------------------------ + wild piece (Q855)
//
// The wolves walk (`designer-case-study-wolves.md` §1 steps 2-3, §3 fix 6) found the wild
// pieces' only way in on a card eleven thousand pixels under the board. The tray's own row
// starts one beside the board, as *+ portals* does: pressed, a type picked under the board
// (one of the draft's pieces, or one out of the library), then a click on the square it
// starts on, which is its walk's first; the card's entry is written and its walk's picker
// opened, so the next clicks are the squares it walks. Only squares clicked are named.

let WILD_TOOL = null;   // { piece: id or '' }: a wild piece being started

// The draft's pieces a wild piece can be: not royal (the validator's rule).
function wildCandidates() {
  return (DRAFT.pieces || []).map((piece, index) => ({ piece, index }))
    .filter(({ piece }) => piece && !piece.royal);
}

// Why *+ wild piece* cannot start one here, or '' when it can.
function wildToolRefused() {
  return draftPortals().length ? "Wild pieces can't be used together with portals yet." : '';
}

function wildRowHtml() {
  const refused = wildToolRefused();
  const on = !!WILD_TOOL;
  return `<div class="row small palette wildRow" data-wildrow>
    <span class="muted trayWho">Wild</span>
    <button data-wildadd class="${on ? 'on' : ''}" aria-pressed="${on}"${refused
      ? ` disabled title="${esc(refused)}"` : ' title="a piece neither player owns, walking'
        + ' squares you pick: choose its type, then click where it starts"'}>+ wild piece</button>
  </div>`;
}

// Under the board while one is being started: which type, and what the next click does.
function wildToolHtml() {
  if (!WILD_TOOL) return '';
  const preview = lastPreview() || {};
  const lattice = (DRAFT.board || {}).lattice || 'square';
  const library = ((DESIGN_KIT && DESIGN_KIT.palette) || [])
    .filter(entry => entry.lattice === lattice)
    .slice().sort((a, b) => String(a.name).localeCompare(String(b.name)));
  const chips = wildCandidates().map(({ piece, index }) => {
    const icon = ((preview.icons || [])[index] || [])[0];
    const picked = WILD_TOOL.piece === piece.id;
    return `<button data-wildtype="${esc(piece.id)}" class="${picked ? 'on' : ''}"
      aria-pressed="${picked}" title="${esc(piece.name || piece.id)}">${icon
        ? `<img src="${esc(icon.href)}" alt="${esc(piece.name || piece.id)}">` : esc(piece.id)}</button>`;
  }).join('');
  const piece = (DRAFT.pieces || []).find(one => one.id === WILD_TOOL.piece);
  const next = piece
    ? `Now click the square the wild <b>${esc(piece.name || piece.id)}</b> starts on: the
       first square of its walk.`
    : 'Pick what it is, then click the square it starts on.';
  return `<div class="small wildTool u-mt-6" data-wildtool>
    <div><b>A wild piece</b> <span class="muted">belongs to neither player and walks the
      squares you pick, round and round.</span> ${next}</div>
    <div class="row small palette wildTypes u-mt-6">${chips}
      ${library.length ? `<select data-wildlib aria-label="a piece out of the library">
        <option value="">or one from the library…</option>
        ${library.map(entry => `<option value="${esc(entry.key)}">${esc(entry.name)}</option>`).join('')}
      </select>` : ''}
      <button data-wildcancel>put away</button></div></div>`;
}

function startWildTool() {
  if (wildToolRefused()) return;
  const types = wildTypes();
  WILD_TOOL = { piece: types.length ? types[types.length - 1] : '' };
  PLACING = null;
  PICKING = null;
  PICK_LINE = '';
  PORTAL_TOOL = null;
  PORTAL_LIFT = null;
  if (DRAFT_MODE !== 'setup') { DRAFT_MODE = 'setup'; TEACHING = null; endBenchGame(''); }
  benchSay('');
}

function endWildTool() {
  if (!WILD_TOOL) return false;
  WILD_TOOL = null;
  return true;
}

// A piece out of the library, taken for a wild one: added to the draft as the library's
// *take one* adds it (its moves are a player's, should the tray put one down), and not
// made the tray's piece, because it is wanted wild.
function wildFromLibrary(key) {
  const entry = ((DESIGN_KIT && DESIGN_KIT.palette) || []).find(one => one.key === key);
  if (!entry || !WILD_TOOL) return;
  const id = freeLetter(entry.id);
  DRAFT.pieces = DRAFT.pieces || [];
  DRAFT.pieces.push(Object.assign(JSON.parse(JSON.stringify(entry.piece)), { id }));
  WILD_TOOL.piece = id;
  drawDesigner();
  checkDraft();
}

// The click on the board: the entry written, its walk one square long, and its picker open.
function wildAt(square) {
  const preview = lastPreview();
  if (!WILD_TOOL || !preview || offBoard(preview, square)) return;
  const piece = (DRAFT.pieces || []).find(one => one.id === WILD_TOOL.piece);
  if (!piece) { benchSay('Pick what the wild piece is first, under the board.'); return; }
  const W = preview.width;
  const name = squareName(square % W, Math.floor(square / W));
  if ((preview.board || [])[square]) {
    benchSay(`A piece already stands on ${name}. A wild piece starts on an empty square.`);
    return;
  }
  const rules = DRAFT.rules = DRAFT.rules || {};
  rules.npcs = Array.isArray(rules.npcs) ? rules.npcs : [];
  const entry = rules.npcs.length;
  rules.npcs.push({ piece: piece.id, route: [name] });
  WILD_TOOL = null;
  const path = ['npcs', entry, 'route'];
  PICKING = Object.assign({ root: 'rules', path }, pickNamed('rules', path, 'its walk'));
  drawDesigner();
  checkDraft();
  drawBoardSide();
  benchSay(`${PICKING.label[0].toUpperCase()}${PICKING.label.slice(1)} starts on ${name}.`
    + ' Click the squares it walks to, in order, and done picking when it is all there.');
}

function wireWildRow(root) {
  if (!root || !root.querySelectorAll) return;
  root.querySelectorAll('[data-wildadd]').forEach(node => {
    node.onclick = () => {
      if (!endWildTool()) startWildTool();
      drawDesigner();
      drawBoardSide();
      // The type is picked under the board, below a tall tray (Q855).
      keepInView(document.querySelector('[data-wildtool]'));
    };
  });
  root.querySelectorAll('[data-wildtype]').forEach(node => {
    node.onclick = () => {
      if (!WILD_TOOL) return;
      WILD_TOOL.piece = node.dataset.wildtype;
      drawBoardSide();
    };
  });
  root.querySelectorAll('[data-wildlib]').forEach(node => {
    node.onchange = () => { if (node.value) wildFromLibrary(node.value); drawBoardSide(); };
  });
  root.querySelectorAll('[data-wildcancel]').forEach(node => {
    node.onclick = () => { endWildTool(); drawBoardSide(); };
  });
}

// A piece picked from the tray takes the board from a square list being picked, as
// starting the picker takes it from the tray (`cards.js`, `data-cpick`): one tool at a
// time, and the click goes where the lit button says. The board goes back to *set up*,
// where the tray is.
function endPickingForTray() {
  const wild = endWildTool();
  if (!PICKING) return wild;
  PICKING = null;
  PICK_LINE = '';
  if (DRAFT_MODE !== 'setup') {
    DRAFT_MODE = 'setup';
    TEACHING = null;
    endBenchGame('');
  }
  return true;
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
      // A piece picked from the tray is the board's tool now, so a pair being put down,
      // an end lifted, or a walk being picked is put away.
      if (endPickingForTray()) {
        PORTAL_TOOL = null;
        PORTAL_LIFT = null;
        drawDesigner();
        drawBoardSide();
        return;
      }
      if (PORTAL_TOOL || PORTAL_LIFT) { PORTAL_TOOL = null; PORTAL_LIFT = null; drawBoardSide(); }
      refreshPalette();
    };
    // Not redrawn here: replacing the node being dragged cancels the drag. The drop's
    // own redraw shows the picker gone.
    node.ondragstart = event => {
      endPickingForTray();
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
    wireWildRow(box);
  });
  document.querySelectorAll('[data-palettenote]').forEach(box => {
    box.innerHTML = paletteNoteBodyHtml();
    wireWildLinks(box);
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
      else if (WILD_TOOL) wildAt(s);
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
  root.querySelectorAll('[data-pickdone]').forEach(node => { node.onclick = donePicking; });
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
