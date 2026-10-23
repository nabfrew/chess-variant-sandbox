// Portals in the designer (P3 of `docs/archive/plan-portals.md`, *The designer's experience* D1-D10;
// groups, shapes and filters since schema 21, `docs/archive/plan-portals-2.md` §5.4, Q843b).
//
// A group of portals is put down the way a piece is: *+ portals* in the tray, then a click on
// each square an end goes on, two or more, then *done* (or Enter). Two ends are a pair, each
// leading to the other; three or more start as a ring, each leading to the next in the order
// they were clicked. Each group then has a chip in the tray and a card in the rules column
// that offers its shape -- a pair, a ring, a funnel -- and, end by end, where it leads, whose
// pieces go in by it and which pieces, and who may move it; and says, in sentences, what it
// all does. The board says what a click did, or why it was refused, in a status line under it
// that a screen reader reads out; the keyboard reaches every square of it and every control.
//
// A group is drawn in its own colour, the same at every end (`--pair-N`, the vortex of
// `docs/archive/plan-portal-look.md` §2.3), so nothing here names an end by a colour: an end is
// named by its square. The file's keys for a pair's two ends, `orange` and `blue`, are data
// and never words (Q838d). A group of two ends is always written as a pair, and a group of
// three or more with `ends` (schema 21's one spelling for one game, `writeGroup`), so the
// validator's *write it as a pair* is never met here.
//
// **Every sentence about the rule is the server's** (`designing.portal_cards`, built from
// the spec's own fields), and every mark about a move is the engine's: where a piece comes
// out of a portal is drawn from the move map (`designing.move_map`'s `transits`), never
// worked out here (rule 3). What this file decides is only which squares the designer's
// own clicks name -- where an end is put, which end is picked up -- the words for that, and
// how the designer's choices are written into the draft.

let PORTAL_TOOL = null,     // { squares } while a new group is being put down
    PORTAL_PICKED = -1,     // the selected group, by its place in `rules.portals`
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
// The most ends across every group (`spec/schema.py` `MAX_PORTAL_ENDS`: the core's eight).
const MAX_PORTAL_ENDS_ALL = 8;

// The four ways through a pair (D4), in the card's order: the spec's word, and which of
// the pair's two ends (its first, its second) a piece may go in at. A way is named by the
// pair's squares (`wayName`).
const PORTAL_WAYS = [
  ['both_ways', [true, true]], ['orange_to_blue', [true, false]],
  ['blue_to_orange', [false, true]], ['closed', [false, false]],
];

// A way's name on its button: *one way: b2 to e6*, the squares the pair stands on.
function wayName(word, pair) {
  const first = esc((pair || {}).orange || '?'), second = esc((pair || {}).blue || '?');
  return { both_ways: 'both ways', orange_to_blue: `one way: ${first} to ${second}`,
           blue_to_orange: `one way: ${second} to ${first}`, closed: 'no way' }[word] || '';
}

// A group as a person calls it: *pair* for two ends, *group* for more, its number, and its
// colour beside it in brackets. The colour is the number's (`--pair-N`), so it is named only
// to say which group is meant. `size` is its number of ends (unknown: a pair).
function pairCalled(index, size) {
  const colour = PORTAL_PAIR_NAMES[index];
  const word = size > 2 ? 'group' : 'pair';
  return colour ? `${word} ${index + 1} (${colour})` : `${word} ${index + 1}`;
}

// *Pair 1* or *Group 2*, for the start of a sentence.
function groupHead(index) {
  const word = groupSize(draftPortals()[index]) > 2 ? 'Group' : 'Pair';
  return `${word} ${index + 1}`;
}

// Who may move an end (D4): the card's choice, and what it writes as `moved_by`.
const PORTAL_MOVERS = [['nobody', []], ['first', [0]], ['second', [1]], ['either', [0, 1]]];

// The move a group is given the first time somebody may move an end: one step in any
// direction, onto an empty square (D4; `designing.PORTAL_STEP`). A group's move leaves
// `mode` out, which means *move only* there (spelling 31, Q1007b).
const PORTAL_STEP = { kind: 'leap', vectors: [[1, 0], [1, 1]], symmetry: 'rotate4' };

// ------------------------------------------------------------------ the draft's portals

function draftPortals() {
  const rules = (DRAFT && DRAFT.rules) || {};
  return Array.isArray(rules.portals) ? rules.portals : [];
}

// A move row's owner (`moveEditor`): a piece's index, or `p<n>` for portal group n, whose
// own `moves` are how its ends move (spelling 31, Q1007b; there is no portal piece).
function isPortalKey(owner) {
  return String(owner).charAt(0) === 'p';
}

function movesOwner(owner) {
  return isPortalKey(owner) ? draftPortals()[Number(String(owner).slice(1))]
    : (DRAFT.pieces || [])[Number(owner)];
}

// The keys a group's ends are addressed by: a pair's `orange` and `blue` (the file's keys),
// a map's places in its `ends`.
function endKeys(group) {
  if (!group) return [];
  return Array.isArray(group.ends) ? group.ends.map((end, place) => place) : ['orange', 'blue'];
}

function groupSize(group) { return endKeys(group).length; }

// Where an end stands, by name, and putting it somewhere else.
function endSquare(group, key) {
  if (!group) return '';
  if (Array.isArray(group.ends)) return ((group.ends[key] || {}).at) || '';
  return group[key] || '';
}

function setEndSquare(group, key, name) {
  if (Array.isArray(group.ends)) group.ends[key].at = name;
  else group[key] = name;
}

// The letter the compiler names an end by (`portal_end_ids`): a pair's `o`/`b` are its keys'
// first letters, a map's `a`, `b`, `c` by place. The payload's `end` field.
function endLetter(group, key) {
  return Array.isArray(group.ends) ? String.fromCharCode(97 + key) : key;
}

// A group's ends, read one way whatever its spelling, as `spec/schema.py`'s `portal_groups`
// reads them: each `{at, to, enter, movedBy, carries}`, `to` the place of the end it leads
// to or null for a way out only, `enter` whose pieces may go in (empty for a way out). A
// pair's two ends lead to each other, an end nobody may go in by leading nowhere.
function groupEnds(group) {
  if (!group) return [];
  if (Array.isArray(group.ends)) {
    const n = group.ends.length;
    return group.ends.map((end, place) => {
      const to = Number.isInteger(end.to) && end.to !== place && end.to >= 0 && end.to < n ? end.to : null;
      const enter = to === null ? [] : Array.isArray(end.enter) ? end.enter.slice().sort() : [0, 1];
      return { at: end.at || '', to, enter, movedBy: (end.moved_by || []).slice().sort(),
               carries: Array.isArray(end.carries) ? end.carries.slice() : null };
    });
  }
  const passage = group.passage || {};
  const opens = [0, 1].map(p => (PORTAL_WAYS.find(([word]) => word === (passage[String(p)] || 'both_ways'))
    || PORTAL_WAYS[0])[1]);
  return ['orange', 'blue'].map((key, side) => {
    const enter = [0, 1].filter(p => opens[p][side]);
    const carried = (group.carries || {})[key];
    return { at: group[key] || '', to: enter.length ? 1 - side : null, enter,
             movedBy: ((group.moved_by || {})[key] || []).slice().sort(),
             carries: Array.isArray(carried) ? carried.slice() : null };
  });
}

// A group written back from its ends: the pair spelling for two (any group of two ends is a
// pair: who goes in at which end is its `passage`, per player), `ends` for three or more,
// every default left out so the draft hashes as the game it is. Keeps the group's own keys
// (its moves, its kind, its bar on going straight back while somebody moves an end). A
// filter is kept only on an end somebody goes in by, since one elsewhere says nothing.
function writeGroup(group, ends) {
  const out = {};
  if (Array.isArray(group.moves) && group.moves.length) out.moves = group.moves;
  if (ends.length === 2) {
    out.orange = ends[0].at;
    out.blue = ends[1].at;
    const passage = {};
    for (const player of [0, 1]) {
      const ins = ends.map(end => end.to !== null && end.enter.includes(player));
      const word = PORTAL_WAYS.find(([, opens]) => opens[0] === ins[0] && opens[1] === ins[1])[0];
      if (word !== 'both_ways') passage[String(player)] = word;
    }
    if (Object.keys(passage).length) out.passage = passage;
    const moved = {}, carries = {};
    ends.forEach((end, side) => {
      const key = ['orange', 'blue'][side];
      if (end.movedBy.length) moved[key] = end.movedBy.slice().sort();
      if (end.carries && end.to !== null && end.enter.length) carries[key] = end.carries.slice();
    });
    if (Object.keys(moved).length) out.moved_by = moved;
    if (Object.keys(carries).length) out.carries = carries;
  } else {
    out.ends = ends.map(end => {
      const one = { at: end.at };
      if (end.to !== null && end.enter.length) {
        one.to = end.to;
        if (!(end.enter.includes(0) && end.enter.includes(1))) one.enter = end.enter.slice().sort();
        if (end.carries) one.carries = end.carries.slice();
      }
      if (end.movedBy.length) one.moved_by = end.movedBy.slice().sort();
      return one;
    });
  }
  if (group.kind !== undefined) out.kind = group.kind;
  if (group.no_straight_back && ends.some(end => end.movedBy.length)) out.no_straight_back = true;
  return out;
}

// The shape of a group's map (`spec/schema.py`'s `portal_map_shape`, read off the draft):
// `pair` (two ends, each leading to the other), `ring` (one cycle through every end),
// `funnel` (one end a way out only, every other leading to it) or `own`.
function groupShape(ends) {
  const n = ends.length, tos = ends.map(end => end.to);
  if (n === 2 && tos[0] === 1 && tos[1] === 0) return 'pair';
  const exits = tos.map((to, place) => (to === null ? place : -1)).filter(place => place >= 0);
  if (n >= 2 && exits.length === 1 && tos.every(to => to === null || to === exits[0])) return 'funnel';
  if (n >= 3 && !exits.length) {
    const seen = new Set();
    let place = 0;
    while (!seen.has(place)) { seen.add(place); place = tos[place]; }
    if (seen.size === n && place === 0) return 'ring';
  }
  return 'own';
}

// The three named shapes (§5.4), each with what it writes: a pair (each leads to the
// other: two ends only), a ring (each leads to the next, in the order placed) and a funnel
// (every end leads to the last one placed, a way out only).
const PORTAL_SHAPES = [
  ['pair', 'a pair', 'each leads to the other'],
  ['ring', 'a ring', 'each leads to the next, in the order placed'],
  ['funnel', 'a funnel', 'all lead to the last one placed, which is only a way out'],
];

function shapeOffered(shape, size) {
  return shape === 'pair' ? size === 2 : shape === 'ring' ? size >= 3 : size >= 2;
}

// The end the spec puts on a square, by name: { pair, end }, or null.
function portalEndNamed(name) {
  const groups = draftPortals();
  for (let pair = 0; pair < groups.length; pair++) {
    for (const end of endKeys(groups[pair])) {
      if (groups[pair] && endSquare(groups[pair], end) === name) return { pair, end };
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

// *b3*, *b3 and e4*, *a3, c3 and e3*.
function squaresSaid(names) { return portalList(names.map(name => name || '?')); }

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

// How many ends the draft's groups have between them.
function portalEndsHeld() {
  return draftPortals().reduce((sum, group) => sum + groupSize(group), 0);
}

// Why *+ portals* cannot add another group here, or '' when it can: the server's answer for
// the draft (`designing.portal_tool`), and the counts for the draft on the page.
function portalToolRefused() {
  if (draftPortals().length >= MAX_PORTAL_PAIRS) return 'A game can have at most four groups of portals.';
  if (portalEndsHeld() + 2 > MAX_PORTAL_ENDS_ALL) return 'A game can have at most eight portals.';
  const tool = ((DRAFT_CHECK || {}).portals || {}).tool;
  return tool && !tool.open ? tool.why : '';
}

// One group's icon as the tray and the card draw it (D3): the vortex the board draws, in
// the group's colour with its number, standing still (no idle turn: a tray or a page of
// cards is never busy). It stands for the whole group, not for any end, so it is the
// plain one -- sunk, both sides' arms -- whatever the ends' own settings; the board and
// the card's ways show those. A fold group's (schema 22) is the pool.
function pairIconHtml(pair, size) {
  const px = size || 26;
  const end = { pair: pair + 1, end: 'orange', movedBy: [], open: [true, true], dest: 0 };
  return `<svg viewBox="0 0 32 32" width="${px}" height="${px}" aria-hidden="true">${
    endGlyphOf(pair, end, 0, 32, { numbered: true })}</svg>`;
}

// One end of group `index` drawn as its kind draws it: the pool for a fold, else the vortex.
function endGlyphOf(index, end, x, cell, options) {
  return (draftPortals()[index] || {}).kind === 'fold'
    ? foldGlyph({ ...end, kind: 'fold' }, x, 0, cell, { plain: true, ...options })
    : portalGlyph(end, x, 0, cell, options);
}

function portalRowHtml() {
  const refused = portalToolRefused();
  const placing = !!PORTAL_TOOL;
  const chips = draftPortals().map((group, index) => {
    const at = esc(squaresSaid(endKeys(group).map(key => endSquare(group, key))));
    const called = pairCalled(index, groupSize(group));
    return `<button data-portalchip="${index}"
      class="portalChip${index === PORTAL_PICKED ? ' on' : ''}"
      aria-pressed="${index === PORTAL_PICKED}" title="${called}: ${at}"
      aria-label="portal ${called}, on ${at}">${pairIconHtml(index)}</button>`;
  }).join('');
  const down = placing ? PORTAL_TOOL.squares.length : 0;
  const done = placing ? `<button data-portaldone${down < 2 ? ' disabled' : ''}
      title="put the group down: two ends are a pair, three or more a ring to start with (Enter)">done</button>` : '';
  return `<div class="row small palette portalRow" data-portalrow>
    <span class="muted trayWho">Portals</span>
    <button data-portaladd draggable="${refused ? 'false' : 'true'}"
      class="${placing ? 'on' : ''}" aria-pressed="${placing}"${refused
        ? ` disabled title="${esc(refused)}"` : ' title="put a group of portals on the board:'
          + ' click a square for each end, two or more, then done"'}>+ portals</button>
    ${done}${chips}
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
  root.querySelectorAll('[data-portaldone]').forEach(node => {
    node.onclick = () => finishPortalTool();
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
  PORTAL_TOOL = { squares: [] };
  PORTAL_LIFT = null;
  PLACING = null;
  PICKING = null;
  WILD_TOOL = null;
  if (DRAFT_MODE !== 'setup') { DRAFT_MODE = 'setup'; TEACHING = null; endBenchGame(''); }
  benchSay(`Click an empty square for the first end of ${pairCalled(draftPortals().length, 3)}.`
    + ' Two ends make a pair; more make a ring or a funnel.');
}

function cancelPortalTool() {
  if (!PORTAL_TOOL) return false;
  PORTAL_TOOL = null;
  benchSay('Put away: no portals were added.');
  drawBoardSide();
  return true;
}

// What the status line says while a group is being put down, after each click.
function placingWords() {
  const names = PORTAL_TOOL.squares.map(boardName);
  const number = draftPortals().length + 1;
  if (names.length === 0) return `Click an empty square for the first end of group ${number}.`;
  if (names.length === 1) return `Click an empty square for the second end of group ${number}.`;
  const room = portalEndsHeld() + names.length < MAX_PORTAL_ENDS_ALL;
  const what = names.length === 2 ? 'a pair' : `${names.length} ends`;
  return `Group ${number} has ends on ${squaresSaid(names)}: ${what}. ${room
    ? 'Click another square to add an end, or press Enter or done to put them down.'
    : 'That is as many portals as a game can have: press Enter or done to put them down.'}`;
}

// *done*: the ends clicked become a group. Two are a pair; three or more a ring.
function finishPortalTool() {
  if (!PORTAL_TOOL) return false;
  const names = PORTAL_TOOL.squares.map(boardName);
  if (names.length < 2) {
    benchSay('A group needs two ends or more: click another empty square.');
    drawBoardSide();
    return true;
  }
  addGroup(names);
  return true;
}

// A group picked from its chip, an end on the board or its card: its ends ringed and its
// map drawn, the other groups dimmed, and its card brought into view (D2).
function pickPair(index, scroll) {
  if (index < 0 || index >= draftPortals().length) return;
  PORTAL_PICKED = index;
  PORTAL_LIFT = null;
  benchSay(`${groupHead(index)} is picked: its ends are ringed, where each leads drawn, any other`
    + ' portals dimmed, and its card says what it does. Click one of its ends to move it, or'
    + ' press Esc.');
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
// the end being moved, so its own square is not refused as another portal's.
function portalRefusal(square, own) {
  const name = boardName(square);
  const held = squareHolds(square);
  if (held.kind === 'absent') return `${name} is not a cell of this board.`;
  if (held.kind === 'wall') return `${name} is a wall: nothing can stand there, a portal included.`;
  // A fold (schema 22, rule 10) may lie under a piece: an end of a fold group being moved
  // may go onto one. A new group is momentum until its card says otherwise, so it may not.
  const folded = own && (draftPortals()[own.pair] || {}).kind === 'fold';
  if (held.kind === 'piece' && !folded) {
    return `${name} has a ${held.name} on it. A portal needs an empty square: pick another, or`
      + ` take the ${held.name} off first.`;
  }
  const there = portalEndNamed(name);
  if (there && !(own && there.pair === own.pair && there.end === own.end)) {
    if (own && own.pair === there.pair) {
      return `That is where another end of this ${groupSize(draftPortals()[own.pair]) > 2 ? 'group' : 'pair'}`
        + ' is. Its ends go on different squares.';
    }
    return `That is an end of ${pairCalled(there.pair, groupSize(draftPortals()[there.pair]))
      .replace(/ \(.*\)$/, '')}. Two portals can't share a square.`;
  }
  return '';
}

// A click on the opening board, for the portals: true when it was theirs. Putting an end of
// a new group down (or taking it back up), moving a lifted end, taking a group off with
// *take off*, or picking an end up.
function portalClick(square) {
  const preview = lastPreview();
  if (!preview) return false;
  const name = boardName(square);
  if (PORTAL_TOOL) {
    const at = PORTAL_TOOL.squares.indexOf(square);
    if (at >= 0) {
      PORTAL_TOOL.squares.splice(at, 1);
      benchSay(`Took the end on ${name} back up. ${placingWords()}`);
      drawBoardSide();
      refreshPortalRow();
      return true;
    }
    const why = portalRefusal(square, null);
    if (why) { benchSay(why, square); drawBoardSide(); return true; }
    if (portalEndsHeld() + PORTAL_TOOL.squares.length >= MAX_PORTAL_ENDS_ALL) {
      benchSay('A game can have at most eight portals: press Enter or done to put these down.', square);
      drawBoardSide();
      return true;
    }
    PORTAL_TOOL.squares.push(square);
    benchSay(placingWords());
    drawBoardSide();
    refreshPortalRow();
    return true;
  }
  const here = portalEndNamed(name);
  if (PORTAL_LIFT) {
    const lifted = PORTAL_LIFT;
    const group = draftPortals()[lifted.pair];
    const at = endSquare(group, lifted.end);
    if (here && here.pair === lifted.pair && here.end === lifted.end) {
      PORTAL_LIFT = null;
      benchSay(`${groupHead(lifted.pair)}'s end on ${at} stays there.`);
      drawBoardSide();
      return true;
    }
    if (here) return liftEnd(here);
    const why = portalRefusal(square, lifted);
    if (why) { benchSay(why, square); drawBoardSide(); return true; }
    setEndSquare(group, lifted.end, name);
    PORTAL_LIFT = null;
    benchSay(`${groupHead(lifted.pair)}'s end that was on ${at} is on ${name} now.`);
    drawDesigner();
    checkDraft();
    drawBoardSide();
    return true;
  }
  if (here) {
    if (PLACING && PLACING.erase) { removePair(here.pair); return true; }
    return liftEnd(here);
  }
  // A click elsewhere puts a picked group down, and does what the tray's tool does.
  if (PORTAL_PICKED >= 0) { PORTAL_PICKED = -1; drawPortalCards(); drawBoardSide(); }
  return false;
}

// The tray's row again (its *done* button follows the ends put down so far).
function refreshPortalRow() {
  if (typeof document === 'undefined' || !document.querySelectorAll) return;
  document.querySelectorAll('[data-portalrow]').forEach(row => {
    const holder = row.parentNode;
    if (!holder || typeof row.outerHTML !== 'string') return;
    row.outerHTML = portalRowHtml();
    wirePortalRow(holder);
  });
}

function liftEnd(end) {
  PORTAL_PICKED = end.pair;
  PORTAL_LIFT = { pair: end.pair, end: end.end };
  const at = endSquare(draftPortals()[end.pair], end.end) || '?';
  benchSay(`Moving ${groupHead(end.pair).toLowerCase()}'s end on ${at}: click an empty square. Click`
    + ' the end again to put it back, or press Esc.');
  drawBoardSide();
  drawPortalCards();
  showPairCard(end.pair);
  return true;
}

// ------------------------------------------------------------------ changing the groups

// A new group on these squares, in the order clicked: two ends as a pair, more as a ring.
// It has no moves while nobody may move an end (spelling 31: a group carries its own).
function addGroup(names) {
  DRAFT.rules = DRAFT.rules || {};
  const group = names.length === 2 ? { orange: names[0], blue: names[1] }
    : { ends: names.map((at, place) => ({ at, to: (place + 1) % names.length })) };
  DRAFT.rules.portals = draftPortals().concat([group]);
  const number = DRAFT.rules.portals.length;
  PORTAL_TOOL = null;
  PORTAL_PICKED = number - 1;
  benchSay(names.length === 2
    ? `Pair ${number} is on ${names[0]} and ${names[1]}, and picked: its ends are ringed and any`
      + ' other portals dimmed until you click elsewhere or press Esc. Its card, among the rules,'
      + ` says what it does: the pair's chip in the tray takes you there.`
    : `Group ${number} is on ${squaresSaid(names)}, a ring, and picked: each end leads to the`
      + ' next in the order you clicked them. Its card, among the rules, offers its shape (a'
      + ' ring, a funnel, or your own map) and says what it does.');
  PORTAL_SAYS_STALE = true;
  drawDesigner();
  checkDraft();
  drawBoardSide();
}

// Kept for the callers that put a pair down by its two squares.
function addPair(orange, blue) { addGroup([orange, blue]); }

// The whole group, every end (D2: one end alone is not a portal); the groups after it move
// up a number, and with the last group the rule goes.
function removePair(index) {
  const groups = draftPortals().slice();
  if (index < 0 || index >= groups.length) return;
  const size = groupSize(groups[index]);
  groups.splice(index, 1);
  if (groups.length) DRAFT.rules.portals = groups;
  else dropPortals();
  PORTAL_LIFT = null;
  PORTAL_PICKED = -1;
  PORTAL_APART = new Set();
  const moved = groups.length > index
    ? ` The ${groupSize(groups[index]) > 2 ? 'group' : 'pair'} that was ${index + 2} is now ${index + 1}${
      groups.length > index + 1 ? ', and the ones after it moved up too' : ''}.` : '';
  benchSay(`Took off ${size > 2 ? `group ${index + 1}, all ${size} ends` : `pair ${index + 1}, both ends`}.`
    + ` Undo brings it back.${moved}`);
  drawDesigner();
  checkDraft();
  drawBoardSide();
}

// No groups left: the rule goes.
function dropPortals() {
  delete DRAFT.rules.portals;
}

// A group's number is its place in the list (D2), so renumbering is moving it there. Its
// colour is its number's (`--pair-N`), so a new number is a new colour too, and the
// sentence says both.
function renumberPair(index, to) {
  const groups = draftPortals().slice();
  if (to < 0 || to >= groups.length || to === index) return;
  const [group] = groups.splice(index, 1);
  groups.splice(to, 0, group);
  DRAFT.rules.portals = groups;
  PORTAL_PICKED = to;
  PORTAL_LIFT = null;
  PORTAL_APART = new Set();
  const other = to < index ? to + 1 : to - 1;
  const word = groupSize(group) > 2 ? 'Group' : 'Pair';
  const otherWord = groupSize(groups[other]) > 2 ? 'group' : 'pair';
  benchSay(`${word} ${index + 1} is now ${pairCalled(to, groupSize(group))}; the ${otherWord} that was ${
    to + 1} is now ${pairCalled(other, groupSize(groups[other]))}. Players see these numbers and colours,`
    + ' so this is a change to the game.');
  drawDesigner();
  checkDraft();
  drawBoardSide();
}

// A card choice made through the ends' one reading: the group written back in its one
// spelling, the cards drawn again and the draft checked.
function changeEnds(index, change) {
  const groups = draftPortals();
  const group = groups[index];
  if (!group) return;
  const ends = groupEnds(group);
  change(ends);
  groups[index] = writeGroup(group, ends);
  DRAFT.rules.portals = groups;
  PORTAL_SAYS_STALE = true;
  drawPortalCards();
  drawBoardSide();
  checkDraft();
}

// One of the three named shapes (§5.4). A pair's two ends lead to each other; a ring's each
// lead to the next in the order placed; a funnel's all lead to the last placed, which is
// only a way out. An end that goes on leading somewhere keeps who may go in by it.
function setShape(index, shape) {
  const group = draftPortals()[index];
  if (!group || !shapeOffered(shape, groupSize(group))) return;
  changeEnds(index, ends => {
    const n = ends.length;
    ends.forEach((end, place) => {
      const to = shape === 'pair' ? 1 - place : shape === 'ring' ? (place + 1) % n
        : place === n - 1 ? null : n - 1;
      end.to = to;
      if (to === null) { end.enter = []; end.carries = null; }
      else if (!end.enter.length) end.enter = [0, 1];
    });
  });
  const name = (PORTAL_SHAPES.find(([key]) => key === shape) || [])[1];
  benchSay(`${groupHead(index)} is ${name} now.`);
}

// Where one end leads (§5.4's *leads to* row): another end of the group, by its place, or
// nowhere (`null`), a way out only. Changing a row may make the shape *your own map*.
function setLeadsTo(index, place, to) {
  changeEnds(index, ends => {
    const end = ends[place];
    if (!end) return;
    end.to = to === null || to === place ? null : to;
    if (end.to === null) { end.enter = []; end.carries = null; }
    else if (!end.enter.length) end.enter = [0, 1];
  });
}

// Whose pieces may go in by one end of a map: both sides', or one side's.
function setEnter(index, place, players) {
  changeEnds(index, ends => {
    const end = ends[place];
    if (end && end.to !== null && players.length) end.enter = players.slice().sort();
  });
}

// The pieces that may go in by one end (§5.2, schema 21's `carries`): all, or the ones
// chosen. Choosing starts with every piece ticked, and the card says to untick the ones it
// keeps out.
function setFilter(index, place, mode) {
  changeEnds(index, ends => {
    const end = ends[place];
    if (!end || end.to === null) return;
    end.carries = mode === 'choose' ? portalFilterPieces().map(piece => piece.id) : null;
  });
}

function toggleCarried(index, place, id) {
  changeEnds(index, ends => {
    const end = ends[place];
    if (!end || !end.carries) return;
    end.carries = end.carries.includes(id) ? end.carries.filter(one => one !== id)
      : portalFilterPieces().map(piece => piece.id).filter(one => one === id || end.carries.includes(one));
  });
}

// The pieces a filter may name: the draft's pieces, in its order.
function portalFilterPieces() {
  return (DRAFT.pieces || []).filter(piece => piece && piece.id);
}

// Which way pieces go through a pair, for one player or (`player` null) for both.
function setPassage(index, player, word) {
  const pair = draftPortals()[index];
  if (!pair) return;
  const passage = Object.assign({}, pair.passage || {});
  if (player === null) { passage['0'] = word; passage['1'] = word; }
  else passage[String(player)] = word;
  for (const key of ['0', '1']) if (passage[key] === 'both_ways') delete passage[key];
  if (Object.keys(passage).length) pair.passage = passage; else delete pair.passage;
  // A filter stays only on an end somebody still goes in by.
  if (pair.carries) {
    const ends = groupEnds(pair);
    ['orange', 'blue'].forEach((key, side) => { if (!ends[side].enter.length) delete pair.carries[key]; });
    if (!Object.keys(pair.carries).length) delete pair.carries;
  }
  PORTAL_SAYS_STALE = true;
  drawPortalCards();
  checkDraft();
}

// Who may move one end. Anybody at all, the first time, gives the group the one step it
// moves by until somebody teaches it another (D4). `end` is a pair's `orange` or `blue`, or
// a map end's place.
function setMovers(index, end, players) {
  const group = draftPortals()[index];
  if (!group) return;
  if (Array.isArray(group.ends)) {
    const one = group.ends[Number(end)];
    if (!one) return;
    if (players.length) one.moved_by = players.slice(); else delete one.moved_by;
    if (!group.ends.some(other => (other.moved_by || []).length)) delete group.no_straight_back;
  } else {
    const movedBy = Object.assign({}, group.moved_by || {});
    if (players.length) movedBy[end] = players.slice(); else delete movedBy[end];
    if (Object.keys(movedBy).length) group.moved_by = movedBy; else delete group.moved_by;
    // Schema 20: a bar on going straight back needs somebody to move the pair.
    if (!group.moved_by) delete group.no_straight_back;
  }
  PORTAL_SAYS_STALE = true;
  let taught = false;
  if (players.length && !(group.moves || []).length) {
    group.moves = [JSON.parse(JSON.stringify(PORTAL_STEP))];
    taught = true;
  }
  // The card's move rows show the step the group was just given, wired with the rest of
  // the form; otherwise only the cards change, and the keyboard's control keeps its focus.
  if (taught) drawDesigner(); else drawPortalCards();
  checkDraft();
}

// Schema 22 (Q843d, `docs/archive/plan-portals-2.md` §5.4): how a group works, *momentum* (a
// piece goes in and is thrown out beside the other end; the default, written by leaving
// `kind` out, as the one spelling of the game) or *fold* (a piece that stops on it may stay, or
// come out of the other end; schema 24).
const PORTAL_KINDS = [
  ['momentum', 'momentum', 'a piece goes in and is thrown out beside the other end'],
  ['fold', 'fold', 'a piece that stops here may stay, or come out of the other end'],
];

function setKind(index, kind) {
  const group = draftPortals()[index];
  if (!group || !PORTAL_KINDS.some(([key]) => key === kind)) return;
  if ((group.kind === 'fold' ? 'fold' : 'momentum') === kind) return;
  if (kind === 'fold') group.kind = 'fold'; else delete group.kind;
  PORTAL_SAYS_STALE = true;
  drawPortalCards();
  drawBoardSide();
  checkDraft();
  benchSay(`${groupHead(index)} is ${kind === 'fold' ? 'a fold' : 'momentum'} now: ${
    PORTAL_KINDS.find(([key]) => key === kind)[2]}.`);
}

// The two kinds as one line of small drawn buttons, each with its glyph as the board draws it
// (the vortex, the pool) and its word, its sentence on it for a hover and a screen reader,
// pressed for the kind the group has. One line, so a card still fits a screen.
function kindChoicesHtml(index, group) {
  const chosen = group.kind === 'fold' ? 'fold' : 'momentum';
  const end = { pair: index + 1, end: 'orange', movedBy: [], open: [true, true], dest: 0 };
  const drawn = kind => `<svg viewBox="0 0 28 28" width="18" height="18" aria-hidden="true">${
    kind === 'fold' ? foldGlyph({ ...end, kind: 'fold' }, 0, 0, 28, { plain: true })
      : portalGlyph(end, 0, 0, 28, { flipped: FLIPPED })}</svg>`;
  return `<div class="kindLine small"><b>How it works:</b><span class="kindChoices" role="group"
    aria-label="how this group works">${PORTAL_KINDS.map(([key, name, says]) => `<button
      data-pkind="${index}.${key}" class="kindChoice${key === chosen ? ' on' : ''}" aria-pressed="${
      key === chosen}" title="${esc(`${name}: ${says}`)}" aria-label="${esc(`${name}: ${says}`)}">${
      drawn(key)}${esc(name)}</button>`).join('')}</span></div>`;
}

// Schema 20 (Q807): whether a portal of the group that has just been moved may be moved
// straight back on the very next move. Offered only while somebody moves an end.
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
  if (endWildTool()) { drawBoardSide(); return true; }
  if (PORTAL_LIFT) {
    const lifted = PORTAL_LIFT;
    PORTAL_LIFT = null;
    benchSay(`${groupHead(lifted.pair)}'s end stays on ${
      endSquare(draftPortals()[lifted.pair], lifted.end) || 'its square'}.`);
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
  const groups = draftPortals();
  const preview = lastPreview();
  marks.pending = [];
  // The ends of a group still being put down: `waiting` draws each as §5's dashed sunk
  // well, an end that is not a portal yet. A group put down that the check has not drawn
  // yet is drawn as the board will draw it.
  if (PORTAL_TOOL) {
    PORTAL_TOOL.squares.forEach((square, place) => {
      marks.pending.push({ square, pair: groups.length + 1, end: String.fromCharCode(97 + place),
                           waiting: true });
    });
  }
  groups.forEach((group, index) => {
    for (const key of endKeys(group)) {
      const square = group ? at(endSquare(group, key)) : -1;
      const drawn = stillPortalAt(preview, square);
      const letter = endLetter(group, key);
      // A fold (schema 22) is drawn from the still's `folds`, not its board.
      const pooled = ((preview && preview.folds) || []).some(fold => fold.square === square
        && fold.pair === index + 1);
      if (square >= 0 && !pooled && !(drawn && drawn.pair === index + 1 && drawn.end === letter)) {
        marks.pending.push({ square, pair: index + 1, end: letter, kind: group.kind });
      }
    }
  });
  if (PORTAL_PICKED >= 0 && groups[PORTAL_PICKED]) {
    const group = groups[PORTAL_PICKED];
    const squares = endKeys(group).map(key => at(endSquare(group, key)));
    marks.ring = new Set(squares);
    // Where each end of the picked group leads, as the play board's group map draws it.
    marks.links = groupEnds(group).filter(end => end.to !== null)
      .map(end => [at(end.at), squares[end.to], PORTAL_PICKED + 1])
      .filter(([a, b]) => a >= 0 && b >= 0);
    marks.dim = new Set();
    groups.forEach((other, index) => {
      if (index !== PORTAL_PICKED && other) {
        for (const key of endKeys(other)) marks.dim.add(at(endSquare(other, key)));
      }
    });
  }
  if (PORTAL_LIFT && groups[PORTAL_LIFT.pair]) {
    marks.lift = at(endSquare(groups[PORTAL_LIFT.pair], PORTAL_LIFT.end));
  }
  return marks;
}

// ------------------------------------------------------------------ the group's card (D4)

// The server's sentences for a group, from the last check (`designing.portal_cards`), found
// by the group's number.
function portalCardSaid(index) {
  const cards = (((DRAFT_CHECK || {}).portals) || {}).cards || [];
  return cards.find(card => card.pair === index + 1) || null;
}

// A way through a pair, drawn as its button shows it: the pair's two ends as the board
// draws them, both in the pair's colour, the first on the left and the second on the
// right (the button's words name their squares), and between them a double arrow, an
// arrow, or a solid block. Each end's arms are the pieces this way lets in there -- both
// sides' where `player` is null, else that player's -- and an end nobody goes in by has
// no arms, a way out, so the button teaches the glyph (D4). Still: no idle turn.
function wayDrawingHtml(word, index, player) {
  const ins = (PORTAL_WAYS.find(([key]) => key === word) || [null, [false, false]])[1];
  const open = goes => [0, 1].map(p => goes && (player === null || player === undefined || p === player));
  const end = (at, goes) => endGlyphOf(index, { pair: index + 1, end: at ? 'blue' : 'orange',
    movedBy: [], open: open(goes), dest: goes ? 0 : null }, at ? 48 : 0, 28, { flipped: FLIPPED });
  const ink = 'stroke="var(--ink)" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"';
  const between = {
    both_ways: `<path d="M30 11h16M42 8l4 3-4 3M34 20l-4-3 4-3M30 17h16" ${ink}/>`,
    orange_to_blue: `<path d="M30 14h16M42 10l4 4-4 4" ${ink}/>`,
    blue_to_orange: `<path d="M46 14h-16M34 10l-4 4 4 4" ${ink}/>`,
    closed: '<rect x="33" y="9" width="10" height="10" fill="var(--ink)"/>',
  }[word] || '';
  return `<svg viewBox="0 0 76 28" width="76" height="28" aria-hidden="true">${
    end(0, ins[0])}${end(1, ins[1])}${between}</svg>`;
}

function wayButtonsHtml(index, player, chosen) {
  const who = player === null ? 'both' : String(player);
  const pair = draftPortals()[index] || {};
  return `<div class="wayChoices" role="group" aria-label="${player === null
      ? 'which way pieces go through' : `which way ${esc(sideWord(player))}'s pieces go through`}">${
    PORTAL_WAYS.map(([word]) => `<button data-pway="${index}.${who}.${word}"
      class="wayChoice${word === chosen ? ' on' : ''}" aria-pressed="${word === chosen}">${
      wayDrawingHtml(word, index, player)}<span>${wayName(word, pair)}</span></button>`).join('')}</div>`;
}

// A shape's small drawing (§5.4): its ends as discs of the group's colour, with arrows where
// each leads -- two facing each other, three round a ring, or two into a third.
function shapeDrawingHtml(shape, index) {
  const hue = `var(--pair-${index + 1})`;
  const dot = (x, y) => `<circle cx="${x}" cy="${y}" r="4.2" fill="${hue}" stroke="var(--portal-edge)" stroke-width="1.2"/>`;
  const ink = 'stroke="var(--ink)" stroke-width="1.4" fill="none" stroke-linecap="round" stroke-linejoin="round"';
  const drawn = {
    pair: `${dot(7, 10)}${dot(37, 10)}<path d="M13 8h18M28 5.5l3 2.5-3 2.5M31 12h-18M16 9.5l-3 2.5 3 2.5" ${ink}/>`,
    ring: `${dot(22, 4.5)}${dot(9, 15.5)}${dot(35, 15.5)}<path d="M17 7l-5 4.5M13 8.5l-1 3 3-.5M15 17h14M26 15l3 2-3 2M33 11l-5-4.5M31 6l-3 .5 1 3" ${ink}/>`,
    funnel: `${dot(6, 4.5)}${dot(6, 15.5)}${dot(37, 10)}<path d="M11 5.5l20 3.5M28 6.3l3 2.7-3.6 1M11 14.5l20-3.5M27.4 10l3.6 1-3 2.7" ${ink}/>`,
  }[shape] || '';
  return `<svg viewBox="0 0 44 20" width="44" height="20" aria-hidden="true">${drawn}</svg>`;
}

// The shapes a group may take, as buttons pressed for the shape it has, and *your own map*
// where its rows make none of them (§5.4).
function shapeChoicesHtml(index, ends) {
  const shape = groupShape(ends);
  const buttons = PORTAL_SHAPES.filter(([key]) => shapeOffered(key, ends.length)).map(([key, name, says]) =>
    `<button data-pshape="${index}.${key}" class="wayChoice shapeChoice${key === shape ? ' on' : ''}"
      aria-pressed="${key === shape}" title="${esc(says)}">${shapeDrawingHtml(key, index)}<span>${
      esc(name)}</span></button>`).join('');
  const own = shape === 'own' ? `<span class="wayChoice shapeChoice on shapeOwn" aria-current="true"><span>your
    own map</span></span>` : '';
  return `<div class="wayChoices shapeChoices" role="group" aria-label="the group's shape">${buttons}${own}</div>`;
}

// One end's row on its group's card (§5.4): *The end on b3 leads to* [the end on e4 / ... /
// nowhere: it is a way out]; on a map's end somebody goes in by, whose pieces go in; on a
// map's end, who may move it (a pair's two say it in their own section, as they always
// have); and which pieces may go in by it. Kept to a few short lines, so a card of eight
// ends still fits a screen.
function leadsRowHtml(index, ends, place) {
  const end = ends[place];
  const called = `the end on ${esc(end.at || '?')}`;
  const mapped = ends.length > 2;
  const options = ends.map((other, at) => at === place ? ''
    : `<option value="${at}"${end.to === at ? ' selected' : ''}>the end on ${esc(other.at || '?')}</option>`).join('')
    + `<option value=""${end.to === null ? ' selected' : ''}>nowhere: it is a way out</option>`;
  const enter = mapped && end.to !== null
    ? `<label class="pick">goes in <select data-penter="${index}.${place}"
        aria-label="whose pieces go in at ${called}">${[['both', "both sides'"], ['0', `${esc(sideWord(0))}'s`],
        ['1', `${esc(sideWord(1))}'s`]].map(([value, words]) => `<option value="${value}"${
        (value === 'both' ? end.enter.length === 2 : end.enter.length === 1 && String(end.enter[0]) === value)
          ? ' selected' : ''}>${words}</option>`).join('')}</select></label>` : '';
  const group = draftPortals()[index] || {};
  const chosen = mapped ? moversOf(group, place) : '';
  const moves = mapped ? `<label class="pick">moved by <select data-pmove="${index}.${place}"
      aria-label="who may move ${called}">${PORTAL_MOVERS.map(([key]) => `<option value="${key}"${
      key === chosen ? ' selected' : ''}>${esc(moverWord(key))}</option>`).join('')}</select></label>` : '';
  return `<div class="leadsRow" data-pend="${index}.${place}"><label class="small">The end on ${
      esc(end.at || '?')} leads to <select data-pleads="${index}.${place}" aria-label="where ${called} leads">${
      options}</select></label>${enter || moves ? `<div class="leadsMore small">${enter}${moves}</div>` : ''}
    ${end.to !== null ? filterHtml(index, ends, place) : ''}</div>`;
}

// Which pieces may go in by one end (§5.2, §5.4): *all* or *choose*, and when choosing, the
// game's pieces as their icons with a tick each, the sentence above them.
function filterHtml(index, ends, place) {
  const end = ends[place];
  const called = `the end on ${esc(end.at || '?')}`;
  const choosing = !!end.carries;
  const preview = lastPreview() || {};
  const pieces = portalFilterPieces();
  const named = choosing ? end.carries.map(id => {
    const piece = pieces.find(one => one.id === id);
    return portalPlural((piece && (piece.name || piece.id)) || id);
  }) : [];
  const said = !choosing ? '' : !named.length ? 'No piece is ticked yet: tick the ones that may go in here.'
    : named.length === pieces.length ? 'Every piece is ticked: untick the ones this end keeps out.'
      : `Only ${portalList(named)} may go in here.`;
  const ticks = choosing ? `<div class="filterSays small" data-pfiltersays="${index}.${place}">${esc(said)}</div>
    <div class="filterPieces" role="group" aria-label="which pieces may go in at ${called}">${pieces.map(piece => {
      const type = (preview.pieceIds || []).indexOf(piece.id);
      const icon = type >= 0 ? ((preview.icons || [])[type] || [])[0] : null;
      const on = end.carries.includes(piece.id);
      return `<button data-pcarry="${index}.${place}.${esc(piece.id)}" class="filterPiece${on ? ' on' : ''}"
        aria-pressed="${on}" aria-label="${esc(piece.name || piece.id)}" title="${esc(piece.name || piece.id)}">${icon
          ? `<img src="${esc(icon.href)}" alt="">` : esc(piece.id)}<span class="filterTick" aria-hidden="true">${
          on ? '&#10003;' : ''}</span></button>`;
    }).join('')}</div>` : '';
  return `<div class="filterRow small"><span>Which pieces may go in here:</span>
    <span role="radiogroup" aria-label="which pieces may go in at ${called}">${['all', 'choose'].map(mode =>
      `<label class="pick"><input type="radio" name="pfilter-${index}-${place}" data-pfilter="${index}.${place}"
        value="${mode}"${(mode === 'choose') === choosing ? ' checked' : ''}> ${mode}</label>`).join('')}</span>
    ${ticks}</div>`;
}

function moversOf(pair, end) {
  const listed = (Array.isArray(pair.ends) ? (pair.ends[end] || {}).moved_by || []
    : (pair.moved_by || {})[end] || []).slice().sort();
  const found = PORTAL_MOVERS.find(([, players]) =>
    JSON.stringify(players) === JSON.stringify(listed));
  return found ? found[0] : 'nobody';
}

function moverWord(key) {
  return { nobody: 'nobody', first: sideWord(0), second: sideWord(1), either: 'either player' }[key];
}

// Who may move one end, named by its square (the file's key for the end is only the
// controls' address).
function moverChoicesHtml(index, pair, end) {
  const chosen = moversOf(pair, end);
  const name = `pmove-${index}-${end}`;
  const at = endSquare(pair, end) || '?';
  const called = `the end on ${esc(at)}`;
  const radios = PORTAL_MOVERS.map(([key]) => `<label class="pick"><input type="radio"
      name="${name}" data-pmove="${index}.${end}" value="${key}"${key === chosen ? ' checked' : ''}>
      ${esc(moverWord(key))}</label>`).join('');
  const menu = `<select data-pmove="${index}.${end}" aria-label="who may move ${called}">${
    PORTAL_MOVERS.map(([key]) => `<option value="${key}"${key === chosen ? ' selected' : ''}>${
      esc(moverWord(key))}</option>`).join('')}</select>`;
  return `<div class="moverRow"><span class="moverEnd">The end on ${esc(at)}:</span>
    <span class="moverRadios" role="radiogroup" aria-label="who may move ${called}">${radios}</span>
    <span class="moverMenu">${menu}</span></div>`;
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
  // A map's sentences, one an end, in the small size, so a card of several ends fits a screen.
  const size = said.spelling === 'ends' ? ' small' : '';
  return `${(said.ways || []).map(line => `<div class="cardSays pairSays${size}">${esc(line)}</div>`).join('')}
    ${(said.filters || []).map(line => `<div class="cardSays pairSays${size}">${esc(line)}</div>`).join('')}
    ${(said.solid || []).map(line => `<div class="muted small cardSays">${esc(line)}</div>`).join('')}
    ${warn}`;
}

function pairMovesSaysHtml(index) {
  const said = portalCardSaid(index);
  if (!said) return '';
  return `<div class="cardSays pairSays${said.spelling === 'ends' ? ' small' : ''}${
    PORTAL_SAYS_STALE ? ' pairStale' : ''}">${esc(said.moves)}</div>`;
}

// The refusals that name this group, in the designer's words, each with *show me* (D7).
function pairErrorsHtml(index) {
  const errors = (((DRAFT_CHECK || {}).portals) || {}).errors || [];
  const mine = errors.filter(error => error.pair === index + 1);
  const group = draftPortals()[index];
  return mine.map(error => {
    const key = Array.isArray((group || {}).ends) ? error.place : error.end;
    const shown = key !== undefined && key !== null;
    return `<div class="error small pairError">${esc(error.says)}${shown
      ? ` <button class="linkish" data-pshow="${index}.${key}">show me</button>` : ''}</div>`;
  }).join('');
}

function pairCardHtml(group, index, count) {
  const ends = groupEnds(group);
  const mapped = Array.isArray(group.ends);
  const keys = endKeys(group);
  const squares = keys.map(key => endSquare(group, key) || '?');
  let ways = '';
  if (!mapped) {
    const passage = group.passage || {};
    const words = ['0', '1'].map(key => passage[key] || 'both_ways');
    const same = words[0] === words[1] && !PORTAL_APART.has(index);
    const buttons = same ? wayButtonsHtml(index, null, words[0])
      : [0, 1].map(player => `<div class="wayFor"><span class="small"><b>${esc(sideWord(player))}</b></span>${
        wayButtonsHtml(index, player, words[player])}</div>`).join('');
    ways = `<b class="small">Which way can pieces go through?</b>
      <label class="pick small"><input type="checkbox" data-psame="${index}"${same ? ' checked' : ''}>
        The same for ${esc(sideWord(0))} and ${esc(sideWord(1))}</label>
      ${buttons}`;
  }
  const order = count > 2 ? `<button data-pmove-to="${index}.${index - 1}"${index ? '' : ' disabled'}
      title="give this group the number before, and that number's colour">up</button><button
      data-pmove-to="${index}.${index + 1}"${index < count - 1 ? '' : ' disabled'}
      title="give this group the number after, and that number's colour">down</button>` : '';
  const word = mapped ? 'group' : 'pair';
  const where = mapped ? squares.map(esc).join(', ') : `${esc(squares[0])} &#10231; ${esc(squares[1])}`;
  const moved = mapped ? group.ends.some(end => (end.moved_by || []).length) : !!group.moved_by;
  return `<div class="pairCard${index === PORTAL_PICKED ? ' on' : ''}" data-paircard="${index}">
    <div class="cardHead pairHead">
      <button class="pairPick" data-ppick="${index}" title="pick this ${word} on the board"
        aria-label="pick portal ${pairCalled(index, ends.length)} on the board">${pairIconHtml(index, 28)}</button>
      <b>Portal ${word} ${index + 1}</b>
      <span class="muted small pairWhere">${where}</span>
      <span class="grow"></span>
      ${index ? `<button data-pfirst="${index}" title="each end shows its group's number and colour
        on the board; this gives this ${word} the number 1, and so ${PORTAL_PAIR_NAMES[0]}, and moves
        the others along">make this ${word} 1</button>` : ''}${order}
      <button data-premove="${index}">remove this ${word}</button>
    </div>
    ${pairErrorsHtml(index)}
    <div class="pairPart">${kindChoicesHtml(index, group)}
      ${shapeChoicesHtml(index, ends)}
      <div class="leadsRows">${ends.map((end, place) => leadsRowHtml(index, ends, place)).join('')}</div>
    </div>
    <div class="pairPart">${ways}
      <div data-psays="${index}">${pairSaysHtml(index)}</div>
    </div>
    <div class="pairPart">${mapped ? '' : `<b class="small">Who may move these portals?</b>
      ${keys.map(key => moverChoicesHtml(index, group, key)).join('')}`}
      ${moved ? `<label class="pick small"><input type="checkbox" data-pback="${index}"${
        group.no_straight_back ? ' checked' : ''}> Not straight back: a portal that has just been
        moved can't be moved back where it came from on the very next move</label>` : ''}
      <div data-pmoves="${index}">${pairMovesSaysHtml(index)}</div>
    </div>
  </div>${groupMovesHtml(group, index, moved)}`;
}

// How the group's portals move when somebody moves one (spelling 31, Q1007b): the group's
// own ways of moving, as a piece card's rows, offering only what a portal may do (D1). Shown
// while somebody may move an end, or while the group still holds a way of moving. A card of
// its own right under the group's card, as the stand-in piece's card was one of its own
// before: the rows are a piece card's and grow with every way added, and inside the group's
// card they pushed it past a 1400 x 900 screen (Q1014), where a picked group's card is to be
// shown whole (the look check's `D1 card`).
function groupMovesHtml(group, index, moved) {
  const moves = Array.isArray(group.moves) ? group.moves : [];
  if (!moved && !moves.length) return '';
  const owner = `p${index}`;
  const word = Array.isArray(group.ends) ? 'group' : 'pair';
  return `<div class="pairMoves" data-pmoverows="${index}" data-count="${moves.length}">
    <div class="pairMovesHead">${pairIconHtml(index, 20)}<b class="small">How a portal of ${word} ${index + 1} moves</b></div>
    ${portalMovesNoteHtml()}
    ${moves.length ? '' : `<div class="muted small">It has no way of moving yet.</div>`}
    ${moves.map((move, m) => moveEditor(move, owner, m)).join('')}
    <div class="row u-mt-7"><button data-addmove="${owner}">add a way of moving</button></div>
  </div>`;
}

// The portals' section of the rules column: a card per group, and the refusals about the
// portals that no one group is at fault for. Records the rule for the form's round
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
  // A group's move rows are the form's own controls: while one is being typed in, the cards
  // wait for the next redraw, so the cursor stays where it is.
  if (active && typeof active.closest === 'function' && active.closest('[data-pmoverows]')) return;
  const pairs = draftPortals();
  box.innerHTML = pairs.map((pair, index) => pairCardHtml(pair, index, pairs.length)).join('');
  wirePortalCards(box);
  if (typeof wireMoveRows === 'function') wireMoveRows(box);
  if (typeof wireCards === 'function') wireCards(box);
  if (!again) return;
  const found = [...box.querySelectorAll(`[${again.attr}="${again.value}"]`)].find(node =>
    node.tagName === again.tag && (again.choice === null || node.value === again.choice));
  if (found && typeof found.focus === 'function') found.focus();
}

// The attributes a card's controls are found again by.
const PORTAL_CONTROLS = ['data-pway', 'data-pmove', 'data-psame', 'data-premove', 'data-pfirst',
  'data-pmove-to', 'data-ppick', 'data-pshow', 'data-pback', 'data-pshape', 'data-pleads',
  'data-penter', 'data-pfilter', 'data-pcarry', 'data-pkind'];

// A map end's key in a control's address is its place, a pair's its `orange` or `blue`.
function endKeyOf(text) {
  return /^\d+$/.test(text) ? Number(text) : text;
}

function wirePortalCards(root) {
  if (!root || !root.querySelectorAll) return;
  root.querySelectorAll('[data-pway]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const [index, who, word] = node.dataset.pway.split('.');
      setPassage(Number(index), who === 'both' ? null : Number(who), word);
    };
  });
  root.querySelectorAll('[data-pkind]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const [index, kind] = node.dataset.pkind.split('.');
      setKind(Number(index), kind);
    };
  });
  root.querySelectorAll('[data-pshape]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const [index, shape] = node.dataset.pshape.split('.');
      setShape(Number(index), shape);
    };
  });
  root.querySelectorAll('[data-pleads]').forEach(node => {
    node.onchange = () => {
      const [index, place] = node.dataset.pleads.split('.').map(Number);
      setLeadsTo(index, place, node.value === '' ? null : Number(node.value));
    };
  });
  root.querySelectorAll('[data-penter]').forEach(node => {
    node.onchange = () => {
      const [index, place] = node.dataset.penter.split('.').map(Number);
      setEnter(index, place, node.value === 'both' ? [0, 1] : [Number(node.value)]);
    };
  });
  root.querySelectorAll('[data-pfilter]').forEach(node => {
    node.onchange = () => {
      const [index, place] = node.dataset.pfilter.split('.').map(Number);
      setFilter(index, place, node.value);
    };
  });
  root.querySelectorAll('[data-pcarry]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const [index, place, ...id] = node.dataset.pcarry.split('.');
      toggleCarried(Number(index), Number(place), id.join('.'));
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
      if (found) setMovers(Number(index), endKeyOf(end), found[1]);
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
      liftEnd({ pair: Number(index), end: endKeyOf(end) });
    };
  });
}

// What a check says about the groups, put on their cards where they stand (D4, D7).
function drawPortalSays() {
  PORTAL_SAYS_STALE = false;
  if (typeof document === 'undefined' || !document.querySelectorAll) return;
  document.querySelectorAll('[data-psays]').forEach(node => {
    node.innerHTML = pairSaysHtml(Number(node.dataset.psays));
  });
  document.querySelectorAll('[data-pmoves]').forEach(node => {
    node.innerHTML = pairMovesSaysHtml(Number(node.dataset.pmoves));
  });
  // The errors and the tool's state change with the check too; both are small.
  const box = document.getElementById ? document.getElementById('dPortalCards') : null;
  if (box && box.querySelectorAll) drawPortalCards();
  refreshPortalRow();
}

// ------------------------------------------------------------------ a group's moves (D1)

// What a group's move rows say above them: why its moves are limited.
function portalMovesNoteHtml() {
  return `<div class="muted small cardSays">Each way is <i>move only</i> (a portal never takes
    anything, and never goes onto another portal) and the same both ways up (a portal has no
    forward of its own).</div>`;
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
  // Schema 24 (`docs/archive/plan-portals-2.md` §5.3R): through a fold, in at one end and out on
  // the end it leads to, where the move ends.
  if (transit.foldIn !== undefined) {
    return `The ${called} goes into the fold on ${boardName(transit.foldIn)} and comes out on ${
      boardName(transit.foldOut)}.`;
  }
  const into = stillPortalAt(preview, transit.portal);
  const out = stillPortalAt(preview, transit.portalOut);
  const way = transit.exitDir;
  if (!into || !out || !(way >= 0 && way < 8)) return '';
  // By number and square, as the play board says it (`portalName`): every end looks alike.
  const pairs = new Set(((preview.portals || {}).ends || []).map(end => end.pair)).size;
  const mapped = portalMapped(into, out);
  return `The ${called} moves ${TRANSIT_GOING[way]} into ${portalName(into, pairs)} on ${
    boardName(transit.portal)}, and comes out one step ${TRANSIT_BESIDE[way]} ${mapped
    ? 'the end it leads to' : 'its partner'} on ${boardName(transit.portalOut)}, on ${boardName(transit.to)}.`;
}

// Where it goes and can come out of one end two ways round, said once (D6).
function choiceLineHtml(map, called) {
  const ways = new Map();
  for (const transit of (map && map.transits) || []) {
    if (transit.portal === undefined) continue;   // a fold's: no choice of where it comes out
    const seen = ways.get(transit.portal) || new Set();
    seen.add(transit.to);
    ways.set(transit.portal, seen);
  }
  for (const [portal, exits] of ways) {
    if (exits.size < 2) continue;
    const end = stillPortalAt(lastPreview(), portal);
    return `<div class="muted small u-mt-6">From ${boardName(map.from)} the ${
      esc(called)} can reach portal ${end ? end.pair : ''} from both sides, so the player chooses
      where it comes out.</div>`;
  }
  return '';
}

// The key, once a sitting, the first time a draft's board has portals (D5).
function portalKeyHtml() {
  if (PORTAL_KEY_SHUT || !(lastPreview() || {}).portals) return '';
  const ends = lastPreview().portals.ends || [];
  // Schema 22 (Q843d): a fold says its own rule, in words, beside the momentum key or alone.
  const fold = ends.some(isFold) ? `<div>A piece that ends its move on a fold may stay there,
    or come out on the end its beak points at, taking a piece there as any move does; a move
    that passes over a fold is not changed.${ends.every(isFold)
      ? ' <button class="linkish" id="dPortalKeyShut">close</button>' : ''}</div>` : '';
  if (ends.length && ends.every(isFold)) return `<div class="portalKey small" id="dPortalKey">${fold}</div>`;
  const arrows = ['&#8598;', '&#8593;', '&#8599;', '&#8592;', '&#9678;', '&#8594;', '&#8601;',
    '&#8595;', '&#8600;'];
  return `<div class="portalKey small" id="dPortalKey">
    <div class="portalKeyGrid" aria-hidden="true">${arrows.map((arrow, at) =>
      `<span${at === 2 ? ' class="on"' : ''}>${arrow}</span>`).join('')}</div>
    <div>A piece comes out beside the end its portal leads to (the one its beak points at),
      one step further in the direction it was going. Up and right, by any amounts, comes out
      up and to the right: a knight and a bishop both come out on the lit corner.
      <button class="linkish" id="dPortalKeyShut">close</button></div>${fold}
  </div>`;
}

// The line under the board while a square is hovered or focused: its transits in words.
function hoverWords(square) {
  const map = DRAFT_MAP;
  if (!map || !map.transits) return '';
  const piece = (DRAFT.pieces || [])[DRAFT_PIECE] || {};
  const called = piece.name || piece.id || 'piece';
  return map.transits.filter(transit => transit.to === square || transit.portal === square
    || (transit.foldIn === square && transit.foldIn !== map.from))
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

// Arrow keys move the keyboard's square, space acts on it as a click would, and so does
// Enter, but for one thing: while a group is being put down with two ends or more, Enter
// is *done*. Esc cancels. The square steps as the board is drawn (the first player at the
// bottom) and over cells the board does not have.
function benchKey(key) {
  const preview = lastPreview();
  if (!preview) return false;
  const W = preview.width, H = preview.height;
  const absent = new Set(preview.absent || []);
  if (key === 'Escape') return benchEscape();
  if (key === 'Enter' && PORTAL_TOOL && PORTAL_TOOL.squares.length >= 2) return finishPortalTool();
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
  if (PICKING) return PICK_LINE ? pickLine(square) : pickSquare(square);
  if (WILD_TOOL) return wildAt(square);
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
  const kept = cells.filter(cell => !absent.has(cell));
  // The line joins from its end nearer the last square the list holds, or the square
  // clicked when it holds none (Q854): `... h4` then row 5 is `h5 ... a5`, the way back
  // along a circuit, where it used to be `a5 ... h5` whichever end the list had reached.
  const held = getIn(rootTarget(PICKING.root), PICKING.path);
  const last = Array.isArray(held) && held.length ? squareIndex(held[held.length - 1], W) : -1;
  const from = last >= 0 && last < W * H ? last : square;
  const apart = cell => Math.max(Math.abs(cell % W - from % W),
                                 Math.abs(Math.floor(cell / W) - Math.floor(from / W)));
  if (kept.length > 1 && apart(kept[kept.length - 1]) < apart(kept[0])) kept.reverse();
  const names = kept.map(boardName);
  const which = PICK_LINE === 'row' ? `row ${rank + 1}` : `column ${String.fromCharCode(97 + file)}`;
  PICK_LINE = '';
  structural(PICKING.root, target => {
    const list = getIn(target, PICKING.path);
    const next = Array.isArray(list) ? list.slice() : [];
    const all = names.every(name => next.includes(name));
    const added = names.filter(name => !next.includes(name));
    setIn(target, PICKING.path, all ? next.filter(name => !names.includes(name))
      : next.concat(added));
    const run = added.length > 1 ? `, ${added[0]} to ${added[added.length - 1]}` : '';
    benchSay(all ? `Took ${which} out of ${PICKING.label}.`
      : `Added ${which} to ${PICKING.label}${run}.`);
  });
  drawBoardSide();
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
  const squares = group => endKeys(group).map(key => endSquare(group, key));
  const key = group => squares(group).join(' ');
  const word = group => (groupSize(group) > 2 ? 'Group' : 'Pair');
  if (now.length > was.length) {
    const back = now.findIndex(pair => !was.some(other => key(other) === key(pair)));
    const at = back < 0 ? now.length - 1 : back;
    return `${word(now[at])} ${at + 1} is back, on ${squaresSaid(squares(now[at]))}.`;
  }
  if (now.length < was.length) {
    const gone = was.findIndex(pair => !now.some(other => key(other) === key(pair)));
    const at = gone < 0 ? was.length - 1 : gone;
    return `${word(was[at])} ${at + 1} is off the board again.`;
  }
  for (let index = 0; index < now.length; index++) {
    if (key(now[index]) !== key(was[index])) {
      return `${word(now[index])} ${index + 1}'s ends are back on ${squaresSaid(squares(now[index]))}.`;
    }
    if (JSON.stringify(now[index]) !== JSON.stringify(was[index])) {
      return `${word(now[index])} ${index + 1}'s card is back as it was.`;
    }
  }
  return '';
}

// Everything here points at a group of the draft on screen, so a draft replaced whole (an
// undo, a template, a design step) leaves none of it standing.
function resetPortalState() {
  PORTAL_TOOL = null;
  PORTAL_LIFT = null;
  PORTAL_PICKED = -1;
  PORTAL_APART = new Set();
  PICK_LINE = '';
  BENCH_FLASH = -1;
  WILD_TOOL = null;
}
