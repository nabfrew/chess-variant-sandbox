// Portals in the designer (P3 of `docs/plan-portals.md`, *The designer's experience* D1-D10).
//
// A pair of portals is put down the way a piece is: *+ portal pair* in the tray, then two
// clicks, orange end then blue. Each pair then has a chip in the tray and a card in the
// rules column that says, in sentences, which way pieces go through it and who may move
// it, with the four ways as named choices that each carry a drawing. The board says what
// a click did, or why it was refused, in a status line under it that a screen reader reads
// out; the keyboard reaches every square of it.
//
// **Every sentence about the rule is the server's** (`designing.portal_cards`, built from
// the spec's own fields), and every mark about a move is the engine's: where a piece comes
// out of a portal is drawn from the move map (`designing.move_map`'s `transits`), never
// worked out here (rule 3). What this file decides is only which squares the designer's
// own clicks name -- where an end is put, which end is picked up -- and the words for that.

let PORTAL_TOOL = null,     // { orange } while a new pair is being put down (-1: none yet)
    PORTAL_PICKED = -1,     // the selected pair, by its place in `rules.portals`
    PORTAL_LIFT = null,     // { pair, end }: an end picked up to be moved
    PORTAL_APART = new Set(),  // the pairs whose card shows a row of ways per player
    PORTAL_KEY_SHUT = false,   // the momentum key, closed for this sitting (D5)
    BENCH_SAID = '',        // the status line under the board
    BENCH_FLASH = -1,       // the square a refused click was on, outlined once
    BENCH_FOCUS = -1,       // the keyboard's square on the bench board (D9)
    BENCH_DRAG = null,      // the end a pointer went down on, for a drag to move it
    PICK_LINE = '',         // 'row' or 'column' while the squares picker takes a whole line
    PORTAL_SAYS_STALE = false;  // a card choice made that the check has not answered yet

const MAX_PORTAL_PAIRS = 4;

// The four ways through a pair (D4), in the card's order: the spec's word, its name.
const PORTAL_WAYS = [
  ['both_ways', 'both ways'], ['orange_to_blue', 'one way: orange to blue'],
  ['blue_to_orange', 'one way: blue to orange'], ['closed', 'no way'],
];

// Who may move an end (D4): the card's choice, and what it writes as `moved_by`.
const PORTAL_MOVERS = [['nobody', []], ['first', [0]], ['second', [1]], ['either', [0, 1]]];

// The move a portal piece is given the first time somebody may move an end: one step in
// any direction, onto an empty square (D4; `designing.PORTAL_STEP`).
const PORTAL_STEP = { kind: 'leap', vectors: [[1, 0], [1, 1]], symmetry: 'rotate4', mode: 'move' };

// ------------------------------------------------------------------ the draft's portals

function draftPortals() {
  const rules = (DRAFT && DRAFT.rules) || {};
  return Array.isArray(rules.portals) ? rules.portals : [];
}

function portalPieceIds() {
  return new Set(draftPortals().map(pair => pair && pair.piece).filter(Boolean));
}

// Whether the piece at this index of the draft is the portal piece, which the tray does
// not offer and the piece list shows last (D1).
function isPortalPiece(index) {
  const piece = ((DRAFT && DRAFT.pieces) || [])[index];
  return !!piece && portalPieceIds().has(piece.id);
}

// The end the spec puts on a square, by name: { pair, end }, or null.
function portalEndNamed(name) {
  const pairs = draftPortals();
  for (let pair = 0; pair < pairs.length; pair++) {
    for (const end of ['orange', 'blue']) {
      if (pairs[pair] && pairs[pair][end] === name) return { pair, end };
    }
  }
  return null;
}

function sideWord(player) {
  const side = ((lastPreview() || {}).sides || ['white', 'black'])[player] || '';
  return side === 'white' || side === 'black' ? side.charAt(0).toUpperCase() + side.slice(1)
    : `the ${side} player`;
}

function boardName(square) {
  const W = (lastPreview() || {}).width || 8;
  return squareName(square % W, Math.floor(square / W));
}

// ------------------------------------------------------------------ the status line

// One sentence under the board, said by a live region a screen reader reads out (D9). The
// element outlives the redraws of the board column, so the reader hears each sentence once.
function benchSay(text, flash) {
  BENCH_SAID = text || '';
  BENCH_FLASH = flash === undefined ? -1 : flash;
  const line = benchSaidNode();
  if (line) line.textContent = BENCH_SAID;
}

function benchSaidNode() {
  if (typeof document === 'undefined' || !document.getElementById) return null;
  let line = document.getElementById('benchSaid');
  if (!line && typeof document.createElement === 'function') {
    line = document.createElement('div');
    if (!line || typeof line.setAttribute !== 'function') return null;
    line.id = 'benchSaid';
    line.className = 'small benchSaid';
    line.setAttribute('role', 'status');
    line.setAttribute('aria-live', 'polite');
  }
  return line;
}

// The status line, put back into the board column after it is drawn again.
function placeBenchSaid(side) {
  const slot = side && side.querySelector ? side.querySelector('[data-saidslot]') : null;
  const line = benchSaidNode();
  if (!slot || !line || typeof slot.replaceWith !== 'function') return;
  line.textContent = BENCH_SAID;
  slot.replaceWith(line);
}

// ------------------------------------------------------------------ the tray's row (D2)

// Why *+ portal pair* cannot add another here, or '' when it can: the server's answer for
// the draft (`designing.portal_tool`), and the count of pairs for the draft on the page.
function portalToolRefused() {
  if (draftPortals().length >= MAX_PORTAL_PAIRS) return 'A game can have at most four pairs of portals.';
  const tool = ((DRAFT_CHECK || {}).portals || {}).tool;
  return tool && !tool.open ? tool.why : '';
}

// One pair's icon as the tray, the card and the board all draw it (D3): its orange end,
// with its number.
function pairIconHtml(pair, size) {
  const px = size || 26;
  const end = { pair: pair + 1, end: 'orange', movedBy: [], open: [true, true] };
  return `<svg viewBox="0 0 32 32" width="${px}" height="${px}" aria-hidden="true">${
    portalGlyph(end, null, null, 0, 0, 32, { numbered: true })}</svg>`;
}

function portalRowHtml() {
  const refused = portalToolRefused();
  const placing = !!PORTAL_TOOL;
  const chips = draftPortals().map((pair, index) => `<button data-portalchip="${index}"
      class="portalChip${index === PORTAL_PICKED ? ' on' : ''}"
      aria-pressed="${index === PORTAL_PICKED}" title="pair ${index + 1}: orange ${
      esc((pair || {}).orange || '?')}, blue ${esc((pair || {}).blue || '?')}"
      aria-label="portal pair ${index + 1}, orange end on ${esc((pair || {}).orange || '?')}, blue end on ${
      esc((pair || {}).blue || '?')}">${pairIconHtml(index)}</button>`).join('');
  return `<div class="row small palette portalRow" data-portalrow>
    <span class="muted trayWho">Portals</span>
    <button data-portaladd draggable="${refused ? 'false' : 'true'}"
      class="${placing ? 'on' : ''}" aria-pressed="${placing}"${refused
        ? ` disabled title="${esc(refused)}"` : ' title="put a pair of portals on the board:'
          + ' click a square for its orange end, then one for its blue end"'}>+ portal pair</button>
    ${chips}
    ${refused ? `<span class="muted portalRefused">${esc(refused)}</span>` : ''}
  </div>`;
}

function wirePortalRow(root) {
  if (!root || !root.querySelectorAll) return;
  root.querySelectorAll('[data-portaladd]').forEach(node => {
    node.onclick = () => togglePortalTool();
    node.ondragstart = event => {
      startPortalTool();
      if (event.dataTransfer) event.dataTransfer.setData('text/plain', 'portal');
    };
  });
  root.querySelectorAll('[data-portalchip]').forEach(node => {
    node.onclick = () => pickPair(Number(node.dataset.portalchip), true);
  });
}

function togglePortalTool() {
  if (PORTAL_TOOL) return cancelPortalTool();
  startPortalTool();
  drawBoardSide();
}

function startPortalTool() {
  if (portalToolRefused()) return;
  PORTAL_TOOL = { orange: -1 };
  PORTAL_LIFT = null;
  PLACING = null;
  PICKING = null;
  if (DRAFT_MODE !== 'setup') { DRAFT_MODE = 'setup'; TEACHING = null; endBenchGame(''); }
  benchSay(`Click an empty square for the orange end of pair ${draftPortals().length + 1}.`);
}

function cancelPortalTool() {
  if (!PORTAL_TOOL) return false;
  PORTAL_TOOL = null;
  benchSay('Put away: no pair was added.');
  drawBoardSide();
  return true;
}

// A pair picked from its chip, an end on the board or its card: its two ends ringed and
// joined, the other pairs dimmed, and its card brought into view (D2).
function pickPair(index, scroll) {
  if (index < 0 || index >= draftPortals().length) return;
  PORTAL_PICKED = index;
  PORTAL_LIFT = null;
  benchSay(`Pair ${index + 1} is picked: its ends are ringed, any other pairs dimmed, and its`
    + ' card says what it does. Click one of its ends to move it, or press Esc.');
  drawBoardSide();
  drawPortalCards();
  if (scroll && PORTAL_PICKED >= 0) showPairCard(PORTAL_PICKED);
}

function showPairCard(index) {
  const card = typeof document !== 'undefined' && document.querySelector
    ? document.querySelector(`[data-paircard="${index}"]`) : null;
  if (card && typeof card.scrollIntoView === 'function') {
    card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

// ------------------------------------------------------------------ a click on the board

// What is on a square of the opening position for a portal to be refused by: a wall, a
// piece, or nothing. Read off the last opening position the check drew, the board the
// person is clicking.
function squareHolds(square) {
  const preview = lastPreview();
  if (!preview) return { kind: 'none' };
  if ((preview.absent || []).includes(square)) return { kind: 'absent' };
  if ((preview.obstacles || []).includes(square)) return { kind: 'wall' };
  const value = (preview.board || [])[square];
  if (!value || stillPortalAt(preview, square)) return { kind: 'none' };
  const type = (value - 1) % preview.nTypes;
  const piece = ((DRAFT.pieces || []).find(one => one.id === (preview.pieceIds || [])[type]));
  return { kind: 'piece', name: (piece && piece.name) || (preview.pieceNames || [])[type] || 'piece' };
}

// Why an end may not go on this square, as the status line says it (D2), or ''. `own` is
// the end being placed or moved, so its own square is not refused as another portal's.
function portalRefusal(square, own) {
  const name = boardName(square);
  const held = squareHolds(square);
  if (held.kind === 'absent') return `${name} is not a cell of this board.`;
  if (held.kind === 'wall') return `${name} is a wall: nothing can stand there, a portal included.`;
  if (held.kind === 'piece') {
    return `${name} has a ${held.name} on it. A portal needs an empty square: pick another, or`
      + ` take the ${held.name} off first.`;
  }
  const there = portalEndNamed(name);
  if (there && !(own && there.pair === own.pair && there.end === own.end)) {
    if (own && own.pair === there.pair) {
      return `That is where the ${there.end} end is. The ${own.end} end goes on a different square.`;
    }
    return `That is pair ${there.pair + 1}'s ${there.end} end. Two portals can't share a square.`;
  }
  if (PORTAL_TOOL && PORTAL_TOOL.orange === square) {
    return 'That is where the orange end is. The blue end goes on a different square.';
  }
  return '';
}

// A click on the opening board, for the portals: true when it was theirs. Placing a new
// pair, moving a lifted end, taking a pair off with *take off*, or picking an end up.
function portalClick(square) {
  const preview = lastPreview();
  if (!preview) return false;
  const name = boardName(square);
  if (PORTAL_TOOL) {
    const pair = draftPortals().length;
    const why = portalRefusal(square, PORTAL_TOOL.orange < 0 ? null : { pair, end: 'blue' });
    if (why) { benchSay(why, square); drawBoardSide(); return true; }
    if (PORTAL_TOOL.orange < 0) {
      PORTAL_TOOL.orange = square;
      benchSay(`Click an empty square for the blue end of pair ${pair + 1}.`);
      drawBoardSide();
      return true;
    }
    addPair(boardName(PORTAL_TOOL.orange), name);
    return true;
  }
  const here = portalEndNamed(name);
  if (PORTAL_LIFT) {
    const lifted = PORTAL_LIFT;
    const at = draftPortals()[lifted.pair][lifted.end];
    if (here && here.pair === lifted.pair && here.end === lifted.end) {
      PORTAL_LIFT = null;
      benchSay(`Pair ${lifted.pair + 1}'s ${lifted.end} end stays on ${at}.`);
      drawBoardSide();
      return true;
    }
    if (here) return liftEnd(here);
    const why = portalRefusal(square, lifted);
    if (why) { benchSay(why, square); drawBoardSide(); return true; }
    draftPortals()[lifted.pair][lifted.end] = name;
    PORTAL_LIFT = null;
    benchSay(`Pair ${lifted.pair + 1}'s ${lifted.end} end is on ${name} now.`);
    drawDesigner();
    checkDraft();
    drawBoardSide();
    return true;
  }
  if (here) {
    if (PLACING && PLACING.erase) { removePair(here.pair); return true; }
    return liftEnd(here);
  }
  // A click elsewhere puts a picked pair down, and does what the tray's tool does.
  if (PORTAL_PICKED >= 0) { PORTAL_PICKED = -1; drawPortalCards(); drawBoardSide(); }
  return false;
}

function liftEnd(end) {
  PORTAL_PICKED = end.pair;
  PORTAL_LIFT = { pair: end.pair, end: end.end };
  benchSay(`Moving pair ${end.pair + 1}'s ${end.end} end: click an empty square. Click the end`
    + ' again to put it back, or press Esc.');
  drawBoardSide();
  drawPortalCards();
  showPairCard(end.pair);
  return true;
}

// ------------------------------------------------------------------ changing the pairs

// The portal piece, made unseen on the first pair (D1): named *portal*, worth nothing, and
// with no move while nobody may move an end. Its letter is one the draft has free and no
// portal end is called. The pairs already there share theirs.
function ensurePortalPiece() {
  const held = draftPortals().find(pair => pair && pair.piece);
  if (held) return held.piece;
  const used = new Set((DRAFT.pieces || []).map(piece => piece.id));
  const id = 'OVWXYZJLMTU'.split('').find(letter => !used.has(letter)) || 'Q';
  DRAFT.pieces = DRAFT.pieces || [];
  DRAFT.pieces.push({ id, name: 'portal', value: 0, moves: [] });
  return id;
}

function addPair(orange, blue) {
  DRAFT.rules = DRAFT.rules || {};
  const piece = ensurePortalPiece();
  DRAFT.rules.portals = draftPortals().concat([{ piece, orange, blue }]);
  const number = DRAFT.rules.portals.length;
  PORTAL_TOOL = null;
  PORTAL_PICKED = number - 1;
  benchSay(`Pair ${number} is on ${orange} and ${blue}, and picked: its ends are ringed and any`
    + ' other pairs dimmed until you click elsewhere or press Esc. Its card, among the rules, says'
    + ` what it does: the pair's chip in the tray takes you there.`);
  PORTAL_SAYS_STALE = true;
  drawDesigner();
  checkDraft();
  drawBoardSide();
}

// The whole pair, both ends (D2: one end alone is not a portal); the pairs after it move
// up a number, and with the last pair the portal piece goes too.
function removePair(index) {
  const pairs = draftPortals().slice();
  if (index < 0 || index >= pairs.length) return;
  pairs.splice(index, 1);
  if (pairs.length) DRAFT.rules.portals = pairs;
  else dropPortals();
  PORTAL_LIFT = null;
  PORTAL_PICKED = -1;
  PORTAL_APART = new Set();
  const moved = pairs.length > index
    ? ` The pair that was ${index + 2} is now ${index + 1}${pairs.length > index + 1
      ? ', and the ones after it moved up too' : ''}.` : '';
  benchSay(`Took off pair ${index + 1}, both ends. Undo brings it back.${moved}`);
  drawDesigner();
  checkDraft();
  drawBoardSide();
}

// No pairs left: the rule goes, and the portal piece no rule names any more.
function dropPortals() {
  const unseen = portalPieceIds();
  delete DRAFT.rules.portals;
  DRAFT.pieces = (DRAFT.pieces || []).filter(piece => !unseen.has(piece.id));
  DRAFT_PIECE = Math.min(DRAFT_PIECE, Math.max(0, DRAFT.pieces.length - 1));
}

// A pair's number is its place in the list (D2), so renumbering is moving it there.
function renumberPair(index, to) {
  const pairs = draftPortals().slice();
  if (to < 0 || to >= pairs.length || to === index) return;
  const [pair] = pairs.splice(index, 1);
  pairs.splice(to, 0, pair);
  DRAFT.rules.portals = pairs;
  PORTAL_PICKED = to;
  PORTAL_LIFT = null;
  PORTAL_APART = new Set();
  benchSay(`Pair ${index + 1} is now pair ${to + 1}; the pair that was ${to + 1} is now ${
    to < index ? to + 2 : to}. Players see these numbers, so this is a change to the game.`);
  drawDesigner();
  checkDraft();
  drawBoardSide();
}

// Which way pieces go through, for one player or (`player` null) for both.
function setPassage(index, player, word) {
  const pair = draftPortals()[index];
  if (!pair) return;
  const passage = Object.assign({}, pair.passage || {});
  if (player === null) { passage['0'] = word; passage['1'] = word; }
  else passage[String(player)] = word;
  for (const key of ['0', '1']) if (passage[key] === 'both_ways') delete passage[key];
  if (Object.keys(passage).length) pair.passage = passage; else delete pair.passage;
  PORTAL_SAYS_STALE = true;
  drawPortalCards();
  checkDraft();
}

// Who may move one end. Anybody at all, the first time, gives the portal piece the one
// step it moves by until somebody teaches it another (D4).
function setMovers(index, end, players) {
  const pair = draftPortals()[index];
  if (!pair) return;
  const movedBy = Object.assign({}, pair.moved_by || {});
  if (players.length) movedBy[end] = players.slice(); else delete movedBy[end];
  if (Object.keys(movedBy).length) pair.moved_by = movedBy; else delete pair.moved_by;
  // Schema 20: a bar on going straight back needs somebody to move the pair.
  if (!pair.moved_by) delete pair.no_straight_back;
  PORTAL_SAYS_STALE = true;
  let taught = false;
  if (players.length) {
    const piece = (DRAFT.pieces || []).find(one => one.id === pair.piece);
    if (piece && !(piece.moves || []).length) {
      piece.moves = [JSON.parse(JSON.stringify(PORTAL_STEP))];
      taught = true;
    }
  }
  // The piece list shows the step the portal piece was just given; otherwise only the
  // cards change, and the control the keyboard is on keeps its focus.
  if (taught) drawDesigner(); else drawPortalCards();
  checkDraft();
}

// Schema 20 (Q807): whether a portal of the pair that has just been moved may be moved
// straight back on the very next move. Offered only while somebody moves the pair.
function setNoStraightBack(index, on) {
  const pair = draftPortals()[index];
  if (!pair) return;
  if (on) pair.no_straight_back = true; else delete pair.no_straight_back;
  PORTAL_SAYS_STALE = true;
  drawPortalCards();
  checkDraft();
}

// Esc on the bench: a placement, a lift, a pick, a whole line, in that order (D9).
function benchEscape() {
  if (PORTAL_TOOL) return cancelPortalTool();
  if (PORTAL_LIFT) {
    const lifted = PORTAL_LIFT;
    PORTAL_LIFT = null;
    benchSay(`Pair ${lifted.pair + 1}'s ${lifted.end} end stays where it was.`);
    drawBoardSide();
    return true;
  }
  if (PICK_LINE) { PICK_LINE = ''; drawBoardSide(); return true; }
  if (PORTAL_PICKED >= 0) {
    PORTAL_PICKED = -1;
    benchSay('Nothing is picked.');
    drawBoardSide();
    drawPortalCards();
    return true;
  }
  return false;
}

// ------------------------------------------------------------------ the board's marks

// The `bench` a setup board is drawn with (`boardPreview`): the portals' state and the
// keyboard's square. On the other tabs only the keyboard's square.
function benchMarks(shown) {
  const marks = { focus: BENCH_FOCUS, flash: BENCH_FLASH };
  if (shown !== 'setup') return marks;
  const W = (lastPreview() || {}).width || 8;
  const at = name => squareIndex(name, W);
  const pairs = draftPortals();
  const preview = lastPreview();
  marks.pending = [];
  if (PORTAL_TOOL && PORTAL_TOOL.orange >= 0) {
    marks.pending.push({ square: PORTAL_TOOL.orange, pair: pairs.length + 1, end: 'orange' });
  }
  pairs.forEach((pair, index) => {
    for (const end of ['orange', 'blue']) {
      const square = pair ? at(pair[end]) : -1;
      const drawn = stillPortalAt(preview, square);
      if (square >= 0 && !(drawn && drawn.pair === index + 1 && drawn.end === end)) {
        marks.pending.push({ square, pair: index + 1, end });
      }
    }
  });
  if (PORTAL_PICKED >= 0 && pairs[PORTAL_PICKED]) {
    const pair = pairs[PORTAL_PICKED];
    marks.ring = new Set([at(pair.orange), at(pair.blue)]);
    marks.link = [at(pair.orange), at(pair.blue)];
    marks.dim = new Set();
    pairs.forEach((other, index) => {
      if (index !== PORTAL_PICKED && other) {
        marks.dim.add(at(other.orange));
        marks.dim.add(at(other.blue));
      }
    });
  }
  if (PORTAL_LIFT && pairs[PORTAL_LIFT.pair]) marks.lift = at(pairs[PORTAL_LIFT.pair][PORTAL_LIFT.end]);
  return marks;
}

// ------------------------------------------------------------------ the pair's card (D4)

// The server's sentences for the pairs, from the last check (`designing.portal_cards`).
function portalCardSaid(index) {
  const cards = (((DRAFT_CHECK || {}).portals) || {}).cards || [];
  return cards[index] || null;
}

// A way through a pair, drawn as its button shows it: the two rims the board draws, the
// blue one double, and between them a double arrow, an arrow, or a solid block.
function wayDrawingHtml(word) {
  const closed = word === 'closed';
  const rim = (cx, hue, blue) => `<circle cx="${cx}" cy="14" r="8" fill="var(--portal-stone)"
      stroke="var(--portal-edge)" stroke-width="5.6"/>
    <circle cx="${cx}" cy="14" r="8" fill="none" stroke="${hue}" stroke-width="4"${
      closed ? ' stroke-dasharray="3 3"' : ''}/>${blue
      ? `<circle cx="${cx}" cy="14" r="8" fill="none" stroke="var(--portal-edge)" stroke-width="0.7"/>` : ''}`;
  const ink = 'stroke="var(--ink)" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"';
  const between = {
    both_ways: `<path d="M24 11h16M36 8l4 3-4 3M28 20l-4-3 4-3M24 17h16" ${ink}/>`,
    orange_to_blue: `<path d="M24 14h16M36 10l4 4-4 4" ${ink}/>`,
    blue_to_orange: `<path d="M40 14h-16M28 10l-4 4 4 4" ${ink}/>`,
    closed: '<rect x="27" y="9" width="10" height="10" fill="var(--ink)"/>',
  }[word] || '';
  return `<svg viewBox="0 0 64 28" width="64" height="28" aria-hidden="true">${
    rim(11, 'var(--portal-a)', false)}${rim(53, 'var(--portal-b)', true)}${between}</svg>`;
}

function wayButtonsHtml(index, player, chosen) {
  const who = player === null ? 'both' : String(player);
  return `<div class="wayChoices" role="group" aria-label="${player === null
      ? 'which way pieces go through' : `which way ${esc(sideWord(player))}'s pieces go through`}">${
    PORTAL_WAYS.map(([word, name]) => `<button data-pway="${index}.${who}.${word}"
      class="wayChoice${word === chosen ? ' on' : ''}" aria-pressed="${word === chosen}">${
      wayDrawingHtml(word)}<span>${esc(name)}</span></button>`).join('')}</div>`;
}

function moversOf(pair, end) {
  const listed = ((pair.moved_by || {})[end] || []).slice().sort();
  const found = PORTAL_MOVERS.find(([, players]) =>
    JSON.stringify(players) === JSON.stringify(listed));
  return found ? found[0] : 'nobody';
}

function moverWord(key) {
  return { nobody: 'nobody', first: sideWord(0), second: sideWord(1), either: 'either player' }[key];
}

function moverChoicesHtml(index, pair, end) {
  const chosen = moversOf(pair, end);
  const name = `pmove-${index}-${end}`;
  const radios = PORTAL_MOVERS.map(([key]) => `<label class="pick"><input type="radio"
      name="${name}" data-pmove="${index}.${end}" value="${key}"${key === chosen ? ' checked' : ''}>
      ${esc(moverWord(key))}</label>`).join('');
  const menu = `<select data-pmove="${index}.${end}" aria-label="who may move the ${end} end">${
    PORTAL_MOVERS.map(([key]) => `<option value="${key}"${key === chosen ? ' selected' : ''}>${
      esc(moverWord(key))}</option>`).join('')}</select>`;
  return `<div class="moverRow"><span class="moverEnd">The ${end} end (${esc(pair[end] || '?')}):</span>
    <span class="moverRadios" role="radiogroup" aria-label="who may move the ${end} end">${radios}</span>
    <span class="moverMenu">${menu}</span></div>`;
}

// How a pair's two ends look to each player, drawn from the compiled payload of the last
// check: the passage chevrons are the player's own (D4: *the card teaches the icon*).
function pairLooksHtml(index) {
  const portals = (lastPreview() || {}).portals;
  if (!portals) return '';
  const ends = portals.ends.filter(end => end.pair === index + 1);
  if (ends.length !== 2) return '';
  const looks = [0, 1].map(player => `<span class="pairLook"><svg viewBox="0 0 64 32" width="64"
      height="32" aria-hidden="true">${ends.map((end, at) => portalGlyph(end,
      portalPartner(portals, end), player, at * 32, 0, 32, { numbered: true })).join('')}</svg>
      <span class="muted small">as ${esc(sideWord(player))} sees them</span></span>`).join('');
  return `<div class="pairLooks"><span class="muted small">How they look:</span>${looks}</div>`;
}

function pairSaysHtml(index) {
  const said = portalCardSaid(index);
  if (!said) return '<div class="muted small">checking&hellip;</div>';
  if (PORTAL_SAYS_STALE) {
    return `<div class="pairStale" aria-busy="true">${(said.ways || []).map(line =>
      `<div class="cardSays pairSays">${esc(line)}</div>`).join('')}<div class="muted small">checking
      the change&hellip;</div></div>`;
  }
  const warn = (said.warnings || []).map(line => `<div class="cardWarn small">${esc(line)}</div>`).join('');
  return `${(said.ways || []).map(line => `<div class="cardSays pairSays">${esc(line)}</div>`).join('')}
    ${(said.solid || []).map(line => `<div class="muted small cardSays">${esc(line)}</div>`).join('')}
    ${warn}`;
}

function pairMovesSaysHtml(index) {
  const said = portalCardSaid(index);
  if (!said) return '';
  return `<div class="cardSays pairSays${PORTAL_SAYS_STALE ? ' pairStale' : ''}">${esc(said.moves)}</div>`;
}

// The refusals that name this pair, in the designer's words, each with *show me* (D7).
function pairErrorsHtml(index) {
  const errors = (((DRAFT_CHECK || {}).portals) || {}).errors || [];
  const mine = errors.filter(error => error.pair === index + 1);
  return mine.map(error => `<div class="error small pairError">${esc(error.says)}${error.end
    ? ` <button class="linkish" data-pshow="${index}.${error.end}">show me</button>` : ''}</div>`).join('');
}

function pairCardHtml(pair, index, count) {
  const passage = pair.passage || {};
  const words = ['0', '1'].map(key => passage[key] || 'both_ways');
  const same = words[0] === words[1] && !PORTAL_APART.has(index);
  const ways = same ? wayButtonsHtml(index, null, words[0])
    : [0, 1].map(player => `<div class="wayFor"><span class="small"><b>${esc(sideWord(player))}</b></span>${
      wayButtonsHtml(index, player, words[player])}</div>`).join('');
  const order = count > 2 ? `<button data-pmove-to="${index}.${index - 1}"${index ? '' : ' disabled'}
      title="give this pair the number before">up</button><button data-pmove-to="${index}.${index + 1}"${
      index < count - 1 ? '' : ' disabled'} title="give this pair the number after">down</button>` : '';
  return `<div class="pairCard${index === PORTAL_PICKED ? ' on' : ''}" data-paircard="${index}">
    <div class="cardHead pairHead">
      <button class="pairPick" data-ppick="${index}" title="pick this pair on the board"
        aria-label="pick portal pair ${index + 1} on the board">${pairIconHtml(index, 28)}</button>
      <b>Portal pair ${index + 1}</b>
      <span class="muted small pairWhere">orange ${esc(pair.orange || '?')} &#10231; blue ${esc(pair.blue || '?')}</span>
      <span class="grow"></span>
      ${index ? `<button data-pfirst="${index}" title="each end shows its pair's number on the board;
        this gives this pair the number 1 and moves the others along">make this pair 1</button>` : ''}${order}
      <button data-premove="${index}">remove this pair</button>
    </div>
    ${pairErrorsHtml(index)}
    <div class="pairPart"><b class="small">Which way can pieces go through?</b>
      <label class="pick small"><input type="checkbox" data-psame="${index}"${same ? ' checked' : ''}>
        The same for ${esc(sideWord(0))} and ${esc(sideWord(1))}</label>
      ${ways}
      <div data-psays="${index}">${pairSaysHtml(index)}</div>
    </div>
    <div class="pairPart"><b class="small">Who may move these portals?</b>
      ${moverChoicesHtml(index, pair, 'orange')}${moverChoicesHtml(index, pair, 'blue')}
      ${pair.moved_by ? `<label class="pick small"><input type="checkbox" data-pback="${index}"${
        pair.no_straight_back ? ' checked' : ''}> Not straight back: a portal that has just been
        moved can't be moved back where it came from on the very next move</label>` : ''}
      <div data-pmoves="${index}">${pairMovesSaysHtml(index)}</div>
    </div>
    <div data-plooks="${index}">${pairLooksHtml(index)}</div>
  </div>`;
}

// The portals' section of the rules column: a card per pair, and the refusals about the
// portal piece that no one pair is at fault for. Records the rule for the form's round
// trip as one held value (`kept`), so an edit to any other rule card writes it back as it
// is rather than dropping it.
function portalCardsHtml(out) {
  const pairs = draftPortals();
  if (!pairs.length) return '';
  if (out) out.push({ path: ['portals'], control: 'kept', raw: pairs, required: false });
  const errors = ((((DRAFT_CHECK || {}).portals) || {}).errors || []).filter(error => !error.pair);
  return `<div class="ruleCard portalCards" data-rule="portals">
    <i hidden ${cattrs('rules', ['portals'], 'kept', false)}></i>
    <div class="cardHead"><b>${esc(ruleTitle('portals', 'portals'))}</b></div>
    ${saysHtml(ruleSays('portals'))}
    ${errors.map(error => `<div class="error small">${esc(error.says)}</div>`).join('')}
    <div id="dPortalCards">${pairs.map((pair, index) => pairCardHtml(pair, index, pairs.length)).join('')}</div>
  </div>`;
}

// The cards again, in place, after a pick or a check: the form around them is not drawn
// again, so nothing being typed into elsewhere loses its cursor.
function drawPortalCards() {
  const box = typeof document !== 'undefined' && document.getElementById
    ? document.getElementById('dPortalCards') : null;
  if (!box || typeof box.querySelectorAll !== 'function') return;
  // The control the keyboard was on comes back focused, so a person tabbing through the
  // card is not sent back to the top of the page by a choice or a check (D9).
  const active = document.activeElement;
  const held = active && typeof box.contains === 'function' && box.contains(active)
    ? PORTAL_CONTROLS.find(attr => active.getAttribute && active.getAttribute(attr) !== null)
    : null;
  const again = held ? { attr: held, value: active.getAttribute(held), tag: active.tagName,
                         choice: active.type === 'radio' ? active.value : null } : null;
  const pairs = draftPortals();
  box.innerHTML = pairs.map((pair, index) => pairCardHtml(pair, index, pairs.length)).join('');
  wirePortalCards(box);
  if (!again) return;
  const found = [...box.querySelectorAll(`[${again.attr}="${again.value}"]`)].find(node =>
    node.tagName === again.tag && (again.choice === null || node.value === again.choice));
  if (found && typeof found.focus === 'function') found.focus();
}

// The attributes a card's controls are found again by.
const PORTAL_CONTROLS = ['data-pway', 'data-pmove', 'data-psame', 'data-premove', 'data-pfirst',
  'data-pmove-to', 'data-ppick', 'data-pshow', 'data-pback'];

function wirePortalCards(root) {
  if (!root || !root.querySelectorAll) return;
  root.querySelectorAll('[data-pway]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const [index, who, word] = node.dataset.pway.split('.');
      setPassage(Number(index), who === 'both' ? null : Number(who), word);
    };
  });
  root.querySelectorAll('[data-psame]').forEach(node => {
    node.onchange = () => {
      const index = Number(node.dataset.psame);
      if (node.checked) {
        PORTAL_APART.delete(index);
        const pair = draftPortals()[index] || {};
        setPassage(index, null, (pair.passage || {})['0'] || 'both_ways');
      } else {
        PORTAL_APART.add(index);
        drawPortalCards();
      }
    };
  });
  root.querySelectorAll('[data-pmove]').forEach(node => {
    node.onchange = () => {
      const [index, end] = node.dataset.pmove.split('.');
      const found = PORTAL_MOVERS.find(([key]) => key === node.value);
      if (found) setMovers(Number(index), end, found[1]);
    };
  });
  root.querySelectorAll('[data-pback]').forEach(node => {
    node.onchange = () => setNoStraightBack(Number(node.dataset.pback), node.checked);
  });
  root.querySelectorAll('[data-premove]').forEach(node => {
    node.onclick = event => { event.preventDefault(); removePair(Number(node.dataset.premove)); };
  });
  root.querySelectorAll('[data-pfirst]').forEach(node => {
    node.onclick = event => { event.preventDefault(); renumberPair(Number(node.dataset.pfirst), 0); };
  });
  root.querySelectorAll('[data-pmove-to]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const [index, to] = node.dataset.pmoveTo.split('.').map(Number);
      renumberPair(index, to);
    };
  });
  root.querySelectorAll('[data-ppick]').forEach(node => {
    node.onclick = event => { event.preventDefault(); pickPair(Number(node.dataset.ppick), false); };
  });
  root.querySelectorAll('[data-pshow]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const [index, end] = node.dataset.pshow.split('.');
      if (DRAFT_MODE !== 'setup') { DRAFT_MODE = 'setup'; TEACHING = null; endBenchGame(''); }
      liftEnd({ pair: Number(index), end });
    };
  });
}

// What a check says about the pairs, put on their cards where they stand (D4, D7).
function drawPortalSays() {
  PORTAL_SAYS_STALE = false;
  if (typeof document === 'undefined' || !document.querySelectorAll) return;
  document.querySelectorAll('[data-psays]').forEach(node => {
    node.innerHTML = pairSaysHtml(Number(node.dataset.psays));
  });
  document.querySelectorAll('[data-pmoves]').forEach(node => {
    node.innerHTML = pairMovesSaysHtml(Number(node.dataset.pmoves));
  });
  document.querySelectorAll('[data-plooks]').forEach(node => {
    node.innerHTML = pairLooksHtml(Number(node.dataset.plooks));
  });
  // The errors and the tool's state change with the check too; both are small.
  const box = document.getElementById ? document.getElementById('dPortalCards') : null;
  if (box && box.querySelectorAll) drawPortalCards();
  document.querySelectorAll('[data-portalrow]').forEach(row => {
    const holder = row.parentNode;
    if (!holder || typeof row.outerHTML !== 'string') return;
    row.outerHTML = portalRowHtml();
    wirePortalRow(holder);
  });
}

// ------------------------------------------------------------------ the portal piece (D1)

// What the portal piece's card says in place of its letter, worth and royal flag: what it
// is for, and why its moves are limited.
function portalPieceNoteHtml() {
  return `<div class="muted small cardSays">Used by the portals: it is never put on the board
    and belongs to neither player. Its ways of moving are how a portal moves when somebody
    may move one, so each is <i>move only</i> (a portal never takes anything, and never
    goes onto another portal) and the same both ways up (a portal has no forward of its
    own).</div>`;
}

// ------------------------------------------------------------------ the exit preview (D5)

// A transit in words, for the line under the board (D5): which way the piece was going,
// which end it goes in by and where it comes out. Every square in it is the engine's.
//
// The way out is the engine's (`exitDir`, `compile.PORTAL_DIRECTIONS` order), and the way
// in is the same direction: momentum is the sign of the move (§2.2).
const TRANSIT_GOING = ['right', 'up and to the right', 'up', 'up and to the left', 'left',
  'down and to the left', 'down', 'down and to the right'];
const TRANSIT_BESIDE = ['to the right of', 'up and to the right of', 'above',
  'up and to the left of', 'to the left of', 'down and to the left of', 'below',
  'down and to the right of'];

function transitWords(transit, called) {
  const preview = lastPreview();
  const into = stillPortalAt(preview, transit.portal);
  const out = stillPortalAt(preview, transit.portalOut);
  const way = transit.exitDir;
  if (!into || !out || !(way >= 0 && way < 8)) return '';
  return `The ${called} moves ${TRANSIT_GOING[way]} into portal ${into.pair}'s ${into.end} end,`
    + ` and comes out one step ${TRANSIT_BESIDE[way]} the ${out.end} end, on ${boardName(transit.to)}.`;
}

// Where it goes and can come out of one end two ways round, said once (D6).
function choiceLineHtml(map, called) {
  const ways = new Map();
  for (const transit of (map && map.transits) || []) {
    const seen = ways.get(transit.portal) || new Set();
    seen.add(transit.to);
    ways.set(transit.portal, seen);
  }
  for (const [portal, exits] of ways) {
    if (exits.size < 2) continue;
    const end = stillPortalAt(lastPreview(), portal);
    return `<div class="muted small" style="margin-top:6px">From ${boardName(map.from)} the ${
      esc(called)} can reach portal ${end ? end.pair : ''} from both sides, so the player chooses
      where it comes out.</div>`;
  }
  return '';
}

// The key, once a sitting, the first time a draft's board has portals (D5).
function portalKeyHtml() {
  if (PORTAL_KEY_SHUT || !(lastPreview() || {}).portals) return '';
  const arrows = ['&#8598;', '&#8593;', '&#8599;', '&#8592;', '&#9678;', '&#8594;', '&#8601;',
    '&#8595;', '&#8600;'];
  return `<div class="portalKey small" id="dPortalKey">
    <div class="portalKeyGrid" aria-hidden="true">${arrows.map((arrow, at) =>
      `<span${at === 2 ? ' class="on"' : ''}>${arrow}</span>`).join('')}</div>
    <div>A piece comes out of the other end one step further in the direction it was going.
      Up and right, by any amounts, comes out up and to the right: a knight and a bishop
      both come out on the lit corner. <button class="linkish" id="dPortalKeyShut">close</button></div>
  </div>`;
}

// The line under the board while a square is hovered or focused: its transits in words.
function hoverWords(square) {
  const map = DRAFT_MAP;
  if (!map || !map.transits) return '';
  const piece = (DRAFT.pieces || [])[DRAFT_PIECE] || {};
  const called = piece.name || piece.id || 'piece';
  return map.transits.filter(transit => transit.to === square || transit.portal === square)
    .map(transit => transitWords(transit, called)).filter(Boolean).join(' ');
}

// Hovering or focusing a square lights the legs that come out on it, or go in by it.
function lightLegs(figure, square) {
  if (!figure || !figure.querySelectorAll) return;
  figure.querySelectorAll('.benchLeg').forEach(leg => {
    const on = square !== null && (Number(leg.dataset.exit) === square
      || Number(leg.dataset.entry) === square);
    leg.classList.toggle('on', on);
  });
  const line = document.getElementById('benchHover');
  if (line) line.textContent = square === null ? '' : hoverWords(square);
}

// ------------------------------------------------------------------ the keyboard (D9)

// Arrow keys move the keyboard's square, Enter and space act on it as a click would, Esc
// cancels. The square steps as the board is drawn (the first player at the bottom) and
// over cells the board does not have.
function benchKey(key) {
  const preview = lastPreview();
  if (!preview) return false;
  const W = preview.width, H = preview.height;
  const absent = new Set(preview.absent || []);
  if (key === 'Escape') return benchEscape();
  if (key === 'Enter' || key === ' ') {
    if (BENCH_FOCUS < 0) return false;
    benchAct(BENCH_FOCUS);
    return true;
  }
  const by = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[key];
  if (!by) return false;
  let at = BENCH_FOCUS;
  if (at < 0) at = Math.floor(H / 2) * W + Math.floor(W / 2);
  else {
    let file = at % W, rank = Math.floor(at / W);
    for (let tries = 0; tries < Math.max(W, H); tries++) {
      file += by[0];
      rank += by[1];
      if (file < 0 || file >= W || rank < 0 || rank >= H) { file -= by[0]; rank -= by[1]; break; }
      if (!absent.has(rank * W + file)) break;
    }
    at = rank * W + file;
  }
  BENCH_FOCUS = at;
  drawBoardSide();
  const line = document.getElementById('benchHover');
  if (line) line.textContent = `${boardName(at)}${hoverWords(at) ? ': ' + hoverWords(at) : ''}`;
  return true;
}

// A click on a bench square, whichever tab is up: what `wireBoards` does with a pointer.
function benchAct(square) {
  const shown = benchShown();
  if (shown === 'teach') return teachClick(square);
  if (PICKING) return pickSquare(square);
  if (shown === 'goes') return askWhereItGoes(square);
  if (portalClick(square)) return;
  if (PLACING) placeAt(square);
}

// ------------------------------------------------------------------ a whole row (D3)

// The squares picker's *a whole row* and *a whole column*: pressed, then a square, and
// the line through it joins the list, or leaves it if all of it is in already. One edit,
// so one undo step. Names only the cells the board has.
function pickLine(square) {
  const preview = lastPreview();
  if (!PICKING || !preview) return;
  const W = preview.width, H = preview.height;
  const absent = new Set(preview.absent || []);
  const file = square % W, rank = Math.floor(square / W);
  const cells = [];
  if (PICK_LINE === 'row') for (let f = 0; f < W; f++) cells.push(rank * W + f);
  else for (let r = 0; r < H; r++) cells.push(r * W + file);
  const names = cells.filter(cell => !absent.has(cell)).map(boardName);
  const which = PICK_LINE === 'row' ? `row ${rank + 1}` : `column ${String.fromCharCode(97 + file)}`;
  PICK_LINE = '';
  structural(PICKING.root, target => {
    const list = getIn(target, PICKING.path);
    const next = Array.isArray(list) ? list.slice() : [];
    const all = names.every(name => next.includes(name));
    const kept = all ? next.filter(name => !names.includes(name))
      : next.concat(names.filter(name => !next.includes(name)));
    setIn(target, PICKING.path, kept);
    benchSay(all ? `Took ${which} out of ${PICKING.label}.` : `Added ${which} to ${PICKING.label}.`);
  });
}

function pickLineHtml() {
  if (!PICKING) return '';
  return `<div class="row small pickLine">
    <button data-pickline="row" class="${PICK_LINE === 'row' ? 'on' : ''}"
      aria-pressed="${PICK_LINE === 'row'}">a whole row</button>
    <button data-pickline="column" class="${PICK_LINE === 'column' ? 'on' : ''}"
      aria-pressed="${PICK_LINE === 'column'}">a whole column</button>
    ${PICK_LINE ? `<span class="muted">Now click a square: its ${PICK_LINE} joins the list, or leaves
      it if all of it is there already.</span>` : ''}
  </div>`;
}

// ------------------------------------------------------------------ undo (D8)

// What an undo or a redo brought back about the portals, in the status line's words.
function portalUndoWords(before, after) {
  const pairsOf = draft => (((draft || {}).rules || {}).portals || []).filter(Boolean);
  const was = pairsOf(before), now = pairsOf(after);
  const key = pair => `${pair.orange} ${pair.blue}`;
  if (now.length > was.length) {
    const back = now.findIndex(pair => !was.some(other => key(other) === key(pair)));
    const at = back < 0 ? now.length - 1 : back;
    return `Pair ${at + 1} is back, on ${now[at].orange} and ${now[at].blue}.`;
  }
  if (now.length < was.length) {
    const gone = was.findIndex(pair => !now.some(other => key(other) === key(pair)));
    return `Pair ${(gone < 0 ? was.length - 1 : gone) + 1} is off the board again.`;
  }
  for (let index = 0; index < now.length; index++) {
    if (key(now[index]) !== key(was[index])) {
      return `Pair ${index + 1}'s ends are back on ${now[index].orange} and ${now[index].blue}.`;
    }
    if (JSON.stringify(now[index]) !== JSON.stringify(was[index])) {
      return `Pair ${index + 1}'s card is back as it was.`;
    }
  }
  return '';
}

// Everything here points at a pair of the draft on screen, so a draft replaced whole (an
// undo, a template, a design step) leaves none of it standing.
function resetPortalState() {
  PORTAL_TOOL = null;
  PORTAL_LIFT = null;
  PORTAL_PICKED = -1;
  PORTAL_APART = new Set();
  PICK_LINE = '';
  BENCH_FLASH = -1;
}
