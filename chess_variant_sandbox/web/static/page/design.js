// Building a game, and building an opponent to play it against.
//
// Two editors over the same idea. A ruleset is data and so is a heuristic player, so both
// can be a form -- and neither form is allowed to know anything about what makes its
// subject legal. Every draft goes to the server, which asks the same validator and the same
// cheap gate the command line asks, and every refusal shown here is one of theirs.
//
// **The board is the answer, not the JSON.** A form full of ``[[1, 2]]`` cannot tell you
// whether what you wrote is the piece you meant, so selecting a piece asks the engine where
// it could go from the middle of an empty board and marks the squares: a dot where it may
// move, a ring where it may take, both where it may do either. That is the diagram every
// book of variants opens with, and it is the single most useful thing this editor draws.
//
// **A game built here is a candidate and nothing more.** It is saved into ``runs/``, which
// git ignores, and climbs the same two rungs as anything a design run proposed. Building a
// game is not a way into the calibration corpus without deciding to put it there.

// --------------------------------------------------------------- building a game

let DRAFT = null,          // the spec being edited
    DRAFT_CHECK = null,    // what the server last said about it
    DRAFT_CHECKED = null,  // ...and when it said it
    DRAFT_PREVIEW = null,  // the last opening position a check could draw
    DRAFT_FROM = '',       // the game it started as, for the lineage
    DRAFT_PARENT = null,   // ...and what that game is: its hash, and what it cost to know
    DRAFT_PIECE = 0,       // which piece the move map is showing
    DRAFT_MAP = null,      // ...and what the engine said about it
    DRAFT_MODE = 'setup',  // what the big board is for: setup, teach, goes
    DRAFT_TIMER = null,
    DRAFT_HASH = '',       // the hash the rungs below were last asked about
    DRAFT_STANDING = null, // ...where that hash is on the rungs, and its family
    DRAFT_SCREEN = null,   // ...what the screen said about it
    DRAFT_SAVED = '',      // the last save's message, kept across redraws of its hash
    DRAFT_UNDO = [],       // whole drafts, oldest first, to go back to
    DRAFT_REDO = [],       // ...and the ones undone, to come forward to
    DRAFT_SNAP = '',       // the draft as it stood at the last snapshot
    STANDING_TIMER = null,
    BOARD_RESHAPED = false, // a board control changed, and the check has not answered yet
    BOARD_REMOVED = '',    // ...what that change took off the board, in words
    DESIGN_KIT = null;     // templates and the opponent vocabulary, fetched once

// How many drafts back a person can go. A draft is a few kilobytes and an experiment is
// worth nothing if the way back is gone, so this is generous rather than exact.
const UNDO_DEPTH = 60;

// Where the draft waits out a reload. `sessionStorage` rather than `localStorage`: the
// draft belongs to this tab and to this sitting, and a draft from last week reappearing
// under an empty editor would be worse than starting from a template.
const DRAFT_KEY = 'chess_variant_sandbox.designer.draft';

// Who plays a draft. Both are rungs of the rated ladder (`web/opponents.LADDER`),
// because a budget is part of an agent's identity and a seat at a budget nobody rated
// is a player nobody measured.
//
// The watch used to seat the weakest searching rung, 400 nodes, which on a draft where
// every piece could turn is not a searching player at all. 45 of the 57 opening moves
// there are turns, the search finished one ply in 198 of 215 moves, and 48% of its moves
// went on turning a piece against random's 85% (Q265, 2026-09-17). The strongest rung
// cheap enough for a watch is 6,400 nodes -- 12 ms a move on that draft and 6-28 ms on
// the heaviest committed rulesets, against 0.24-0.67 s for the rung above it, a whole
// unattended game of which would cost the shared box a core.
const WATCH_AGENT = 'minimax:nodes=6400';
const PLAY_AGENT = 'minimax:nodes=1600';

async function designKit() {
  if (!DESIGN_KIT) DESIGN_KIT = await api('/api/design');
  return DESIGN_KIT;
}

async function showDesigner(from, rule) {
  stopPolling();
  // A game that was in hand went with the markup it was drawn into, whether this is a
  // return from the board page or a fresh editor.
  endBenchGame('');
  SESSION = null;
  // A piece added to some earlier draft is not news about this one.
  FRESH_PIECE = null;
  // The preview shares `squareXY` with the board, and `squareXY` reads FLIPPED. A
  // designer opened after playing the second seat of a game would otherwise draw its
  // own board upside down.
  FLIPPED = false;
  // The library too, for *start again from* and for a fork's new name. A link straight
  // to the designer never passed through the listing, so it is asked for here.
  const [kit] = await Promise.all([
    designKit(), LIBRARY.length ? null : roster().catch(() => null),
  ]);
  // LP7: `#/design?rule=<key>`, a door at one rule (`openDoor`); its sentence is said once.
  DOOR_SAID = null;
  const door = typeof rule === 'string' ? await openDoor(kit, rule) : null;
  if (door || DOOR_SAID) {
    // The door made the draft, or said there was none and left the one in hand.
    if (!DRAFT) takeDraft(heldDraft(), kit.templates[0].spec);
  } else if (from && (from !== DRAFT_FROM || AFRESH)) {
    AFRESH = false;
    const source = await api('/api/design/' + from);
    DRAFT = forkOf(source, from);
    resetPortalState();
    DRAFT_FROM = from;
    // What the game you started from *is*, so the editor can say what changing a rule
    // costs before anything has been changed rather than after it has been saved.
    DRAFT_PARENT = source;
    DRAFT_PIECE = 0;
    keepDraft();
    // The address goes back to the plain designer once the copy is made. Left on
    // `#/design/<hash>`, a reload (which a phone does whenever it restores a tab) made
    // the copy again over every edit since, and the held draft was never read.
    try { history.replaceState(null, '', '#/design'); } catch (error) { /* stays */ }
  } else if (!DRAFT) {
    // A draft the page still holds is the one being edited; otherwise the one this tab
    // was editing before it was reloaded, and a template only when there is neither.
    takeDraft(heldDraft(), kit.templates[0].spec);
  }
  wireUndoKeys();
  wireFootbar();

  view.innerHTML = `
    <button class="crumb" id="back">&larr; all games</button>
    <h2 class="u-mt-10">Build a game</h2>
    <p class="muted small u-maxw-70ch">
      A game here is a board, some pieces, where they start and how it ends. Anything
      you build can be played at once and saved as your draft.
      ${DRAFT_FROM ? `Started from <b>${esc(
        (DRAFT_PARENT && DRAFT_PARENT.name) || nameOf(DRAFT_FROM) || DRAFT_FROM)}</b>.` : ''}
    </p>
    ${doorSaidHtml()}
    ${phoneNoteHtml()}
    ${startFromHtml(kit)}
    <div class="designer bench u-mt-12">
      <div id="boardSide" class="stick"></div>
      <div id="form"></div>
      <div id="verdictSide" class="stick"></div>
    </div>`;
  document.getElementById('back').onclick = () => go('#/');
  const noted = document.getElementById('dPhoneNote');
  if (noted) {
    noted.querySelector('button').onclick = () => { noted.remove(); };
  }
  wireStartFrom();
  wireRuleFile();
  wireSticky();
  drawDesigner();
  drawBoardSide();
  checkDraft();
  loadMoveMap();
  // Whether this copy serves the browser build, which is what measuring here needs.
  hereLookup();
  doorDrawn(door);
}

// ------------------------------------------------ a door at a rule (LP7)
//
// `#/design?rule=<key>` (`docs/plan-landing.md` §5, box LP7): the designer opened on a copy of
// chess with one rule added, from a word of the landing's wall or a story's card. Which of the
// designer's own add paths a key takes is the server's (`designing.doors`, in the kit): *add a
// rule* for a rule's card, *how the players take turns* for rounds and several moves a turn,
// *add a way of moving*, a line of sight or a piece's field on the piece the server names, and
// *the cells are* hexagons. So the card arrives holding what the designer would give it, and
// the check judges the draft as it judges any other. A rule chess cannot take is tried on the
// templates, as the server's refusals say (`POST /api/design/rules`). A key with no door
// opens the designer as it was, and says so.
let DOOR_SAID = null;   // what the last door did, said once under the heading

async function openDoor(kit, key) {
  const all = (kit.doors && kit.doors.doors) || {};
  const door = key && Object.prototype.hasOwnProperty.call(all, key) ? all[key] : null;
  try { history.replaceState(null, '', '#/design'); } catch (error) { /* stays */ }
  if (!door) {
    DOOR_SAID = { missing: key };
    return null;
  }
  // The draft in hand is kept for undo, as a template chosen over it is.
  if (DRAFT) snapshotDraft();
  const from = await doorStart(kit);
  let how = { from };
  if (door.kind === 'rule') how = await doorRule(kit, door, from);
  else if (door.kind !== 'board' && (door.kind !== 'turns' || door.field)) doorPiece(door);
  keepDraft();
  DOOR_SAID = Object.assign({ said: door.said, withSaid: door.withSaid }, how);
  return door;
}

// The copy a door starts from: the server's (chess), else the first template.
async function doorStart(kit) {
  const start = kit.doors && kit.doors.start;
  const source = start && start.hash
    ? await api('/api/design/' + start.hash).catch(() => null) : null;
  resetPortalState();
  DRAFT_PIECE = 0;
  if (source) {
    DRAFT = forkOf(source, start.hash);
    DRAFT_FROM = start.hash;
    DRAFT_PARENT = source;
    return source.name || start.name;
  }
  takeDraft(null, kit.templates[0].spec);
  return kit.templates[0].title;
}

// *Add a rule*'s card for `key` on the draft in hand: `fieldDefault`, and its companion.
// Where the card's default cannot be a game (a portal's ends, Q1036), the door carries
// what the server chose instead (`card`), and the rules the card needs beside it (`with`).
function doorAddRule(key, door) {
  const model = cardModel();
  const field = model && model.rules.find(f => f.key === key);
  if (!field) return null;
  DRAFT.rules = DRAFT.rules || {};
  Object.entries((door && door.with) || {}).forEach(([other, value]) => {
    if (!(other in DRAFT.rules)) DRAFT.rules[other] = JSON.parse(JSON.stringify(value));
  });
  DRAFT.rules[key] = door && door.card !== undefined
    ? JSON.parse(JSON.stringify(door.card)) : fieldDefault(field, [key], 'rules');
  const companion = RULE_COMPANIONS[key];
  if (companion) companion(DRAFT.rules);
  return DRAFT.rules[key];
}

// A rule the start already has is its card, as it stands. One it cannot take is tried on
// each template in turn; where none takes it either, it goes on the start, and the check
// says why it does not stand.
async function doorRule(kit, door, from) {
  if (door.rule in (DRAFT.rules || {})) return { from, present: true };
  const start = { draft: JSON.stringify(DRAFT), from: DRAFT_FROM, parent: DRAFT_PARENT, said: from };
  const takes = async () => {
    const before = JSON.parse(JSON.stringify(DRAFT));
    const card = doorAddRule(door.rule, door);
    const answer = await api('/api/design/rules',
      { spec: before, candidates: { [door.rule]: card } }).catch(() => null);
    if (answer && answer.refused && answer.refused[door.rule]) {
      DRAFT = before;
      return false;
    }
    return true;
  };
  if (await takes()) return { from };
  for (const template of kit.templates || []) {
    takeDraft(null, template.spec);
    if (await takes()) return { from: template.title, instead: from };
  }
  DRAFT = JSON.parse(start.draft);
  DRAFT_FROM = start.from;
  DRAFT_PARENT = start.parent;
  doorAddRule(door.rule, door);
  return { from: start.said };
}

// A door that adds to one piece: the piece the server names, opened (PF4's fold), with a
// way of moving of the door's kind, a line of sight, or the field, as the card's own
// buttons and fields add them.
function doorPiece(door) {
  const pieces = DRAFT.pieces || [];
  const index = Math.max(0, Math.min(door.piece || 0, pieces.length - 1));
  const piece = pieces[index];
  if (!piece) return;
  DRAFT_PIECE = index;
  PIECES_OPEN.add(piece.id);
  if (door.kind === 'move') {
    const move = { kind: 'slide', vectors: [[0, 1]], symmetry: 'rotate4' };
    writeMoveField(move, 'kind', door.move);
    Object.assign(move, JSON.parse(JSON.stringify(door.shape || {})));
    piece.moves = (piece.moves || []).concat([move]);
  } else if (door.kind === 'sight') {
    piece.sight = (Array.isArray(piece.sight) ? piece.sight : [])
      .concat([JSON.parse(JSON.stringify(NEW_SIGHT))]);
  } else if (door.kind === 'piece') {
    const field = ((cardModel() || {}).piece || []).find(f => f.key === door.field);
    if (field && piece[field.key] === undefined) {
      // The server's choice where the field's default is no game (Q1036: squares to keep to).
      piece[field.key] = door.value !== undefined ? JSON.parse(JSON.stringify(door.value))
        : fieldDefault(field, [field.key], 'piece.' + index);
    }
  }
}

// Once the designer is drawn: the two doors that are a control's own answer (rounds or
// several moves a turn; hexagons) press it, and the card the door opened on is brought
// into view.
function doorDrawn(door) {
  if (!door) return;
  const shown = () => {
    const card = doorCard(door);
    if (!card) return;
    card.classList.add('doorCard');
    if (typeof card.scrollIntoView === 'function') card.scrollIntoView({ block: 'center' });
  };
  if (door.kind === 'turns') {
    const asked = chooseTurnWay(door.way);
    return asked && asked.then ? asked.then(shown) : shown();
  }
  if (door.kind === 'board') {
    const lattice = document.getElementById('dLattice');
    if (lattice && lattice.value !== door.lattice) {
      lattice.value = door.lattice;
      const asked = lattice.onchange && lattice.onchange();
      return asked && asked.then ? asked.then(shown) : shown();
    }
  }
  shown();
}

function doorCard(door) {
  if (door.kind === 'rule') return document.querySelector(`.ruleCard[data-rule="${door.rule}"]`);
  if (door.kind === 'board') return document.getElementById('dLattice');
  if (door.kind === 'turns' && !door.field) return document.getElementById('dTurns');
  return document.querySelector(`.editorPiece[data-piece="${DRAFT_PIECE}"]`);
}

function doorSaidHtml() {
  const said = DOOR_SAID;
  if (!said) return '';
  if (said.missing !== undefined) {
    return `<p class="small doorSaid" role="status">${said.missing
      ? `There is no rule called <b>${esc(said.missing)}</b> to start a game with`
      : 'No rule was named to start a game with'}, so the designer opens as it was. Every
      rule it has is under <i>add a rule</i>, below the pieces.</p>`;
  }
  const start = said.instead
    ? `<b>${esc(said.said)}</b> does not go in ${esc(said.instead)} as it stands, so this is
      the template <b>${esc(said.from)}</b> with it added`
    : said.present
      ? `A copy of <b>${esc(said.from)}</b>, which already has <b>${esc(said.said)}</b>`
      : `A copy of <b>${esc(said.from)}</b> with <b>${esc(said.said)}</b> added`;
  const beside = said.withSaid && said.withSaid.length && !said.present
    ? `, and <b>${said.withSaid.map(esc).join('</b> and <b>')}</b>, which it needs` : '';
  return `<p class="small doorSaid" role="status">${start}${beside}: its card is marked below.
    Change anything, and the draft is checked as you go.</p>`;
}

// F11 (`first-visit-study.md`): the wheel over a sticking column scrolled the column, not
// the page. The verdict column is longer than a screen, so it had a scrollbar of its own
// and every turn of the wheel over it went there; on a wide screen a thousand pixels of
// wheel moved the page not at all. Now neither column scrolls on its own. A column that
// fits the window sticks by its top, as before; one taller than the window sticks by its
// foot, which is the offset set here: it travels with the page until its last line is on
// the screen, and stays there. The stylesheet says where the columns stick at all, and
// this reads it back (`position`), so the width is written there alone.
const STICK_GAP = 12;

function fitSticky() {
  document.querySelectorAll('.designer .stick').forEach(column => {
    const sticks = window.getComputedStyle(column).position === 'sticky';
    const room = window.innerHeight - pickReach(column) - STICK_GAP;
    column.style.setProperty('--stick-top',
      (sticks ? Math.min(STICK_GAP, room) : STICK_GAP) + 'px');
  });
}

// How much of a column has to be on the screen. All of it, except the board's column while
// squares are picked (Q854, the wolves walk step 6): then the board and the pick note under
// it, so a column taller than the window sticks with the board in view instead of by its
// foot, which left the board off the top while the card that started the pick was in view.
function pickReach(column) {
  const note = PICKING && column.id === 'boardSide' ? column.querySelector('.pickLine') : null;
  if (!note) return column.offsetHeight;
  return Math.ceil(note.getBoundingClientRect().bottom - column.getBoundingClientRect().top);
}

// The height the phone's foot bar takes off the bottom of the screen, or 0 where there is
// none (it is in the verdict's panel there).
function footbarRoom() {
  const bar = document.getElementById('dActions');
  return bar && footbarFixed() ? bar.offsetHeight : 0;
}

// Scrolls the page by as little as brings `node` whole onto the screen, above the phone's
// foot bar; nothing where it is already.
function keepInView(node) {
  if (!node || typeof node.getBoundingClientRect !== 'function') return;
  const box = node.getBoundingClientRect(), room = window.innerHeight - footbarRoom();
  if (box.bottom > room) window.scrollBy(0, Math.min(box.bottom - room + 8, box.top - 8));
  else if (box.top < 0) window.scrollBy(0, box.top - 8);
}

// *pick on the board* pressed on a card: the board, the pick note and its *a whole row*
// are brought into view (Q854). Where the board's column sticks, the column sticks by its
// top while the pick lasts (`pickReach`) and nothing scrolls; stacked, on a phone, the page
// scrolls up to the board, from a card that can be eleven thousand pixels down the form.
function boardToPick() {
  fitSticky();
  const figure = document.querySelector('#boardSide figure.benchBoard');
  const line = document.querySelector('#boardSide .pickLine');
  if (!figure || typeof figure.getBoundingClientRect !== 'function') return;
  const top = figure.getBoundingClientRect().top;
  const bottom = (line || figure).getBoundingClientRect().bottom;
  if (top >= 0 && bottom <= window.innerHeight - footbarRoom()) return;
  figure.scrollIntoView({ block: 'start' });
}

function wireSticky() {
  fitSticky();
  if (!wireSticky.done) {
    wireSticky.done = true;
    window.addEventListener('resize', fitSticky);
  }
  // A column changes height on every check, a tab, an opened panel: each is a new offset.
  if (typeof ResizeObserver === 'function') {
    if (wireSticky.watch) wireSticky.watch.disconnect();
    wireSticky.watch = new ResizeObserver(fitSticky);
    document.querySelectorAll('.designer .stick').forEach(column =>
      wireSticky.watch.observe(column));
  }
}

// Said once, on a phone (Q545; the user, 2026-10-01: *"for the designer in particular,
// desktop should be considered the primary interface, and can add a note that the
// experience is better on desktop"*). Once is this browser's: the first designer opened
// on a narrow screen says it and remembers that it has. Storage refused or absent says it
// every time, which is the lesser fault.
const PHONE_NOTE_KEY = 'designerPhoneNoteSaid';

function phoneNoteHtml() {
  const narrow = typeof window.matchMedia === 'function'
    && window.matchMedia('(max-width: 760px)').matches;
  if (!narrow) return '';
  try {
    if (window.localStorage.getItem(PHONE_NOTE_KEY)) return '';
    window.localStorage.setItem(PHONE_NOTE_KEY, '1');
  } catch (error) { /* said every time, then */ }
  return `<div class="phoneNote small muted" id="dPhoneNote">
    <span>Building a game works on a phone, but it is easier on a computer: the board, the
      pieces and what the game says all fit side by side there.</span>
    <button>got it</button></div>`;
}

// ------------------------------------------------------------- the board in the middle
//
// One large board, and everything a person does to the game happens on it. The three
// modes are three different questions, so each gets the board it needs rather than a
// drawing that half answers all three:
//
// *set up* is the opening position, and the tray beside it puts pieces down on it.
// *teach a move* is the chosen piece alone, because what a descriptor means is a fact
// about an empty board and a drawing among other pieces cannot show it.
// *where it goes* is the opening position again, with the engine's legal moves for the
// piece where it actually stands -- blockers, its own army and every other rule included.
const BENCH_MODES = [
  ['setup', 'set up', 'put the pieces where they start'],
  ['teach', 'teach a move', 'click where this piece should be able to go'],
  ['goes', 'where it goes', 'what this piece may do from the opening position'],
  ['play', 'play it here', 'play the draft on this board, beside the editor, against the'
    + ' machine or by hand. No move list: the buttons at the foot open it on its own page'],
];

// Two ways of playing the draft, and each says which it is (Q532, a newcomer walking the
// designer: *"two buttons say play it, and nothing says which is which"*). The tab above
// plays on this board, beside the editor, with no move list; the verdict's two buttons
// open the board page, full size and with the move list. The words are here once so the
// invitation under the tab can name the other pair by what they say.
const PLAY_AWAY = 'play it on its own page', WATCH_AWAY = 'watch it on its own page';

// A button's words with what the two share set apart, so the bar at the foot of a phone
// says *on its own page* once, before the pair, and fits *save it* and both on one row
// (Q801: three buttons on two rows took 181 of 812 px). The button's name stays whole
// for whoever cannot see the bar, and in the panel of a wide screen nothing changes.
const AWAY_TAIL = ' on its own page';

function awayLabel(words) {
  const at = words.indexOf(AWAY_TAIL);
  return at < 0 ? esc(words)
    : `${esc(words.slice(0, at))}<span class="awayTail">${esc(AWAY_TAIL)}</span>`;
}

function benchMode(mode) {
  if (DRAFT_MODE === mode) return;
  DRAFT_MODE = mode;
  // A move being taught belongs to the mode it was begun in.
  if (mode !== 'teach') TEACHING = null;
  if (mode !== 'setup') WILD_TOOL = null;
  // The editor is asking for the board back, so the game standing on it is over. Said
  // with no sentence: the person pressed the tab, and they know what they pressed.
  if (mode !== 'play') endBenchGame('');
  drawBoardSide();
  // A played board asks the engine per move and needs no map of what one piece means.
  if (mode !== 'play') loadMoveMap();
}

function benchTabsHtml() {
  return BENCH_MODES.map(([mode, label, says]) =>
    `<button data-bench="${mode}" class="${DRAFT_MODE === mode ? 'on' : ''}"
      title="${esc(says)}">${esc(label)}</button>`).join('');
}

// Which board is actually drawn. A rule card asking for squares wants them picked on the
// opening position, and it asks from the column beside this one -- so a pick takes the
// board back to the setup position for as long as it lasts, whatever tab is lit.
function benchShown() {
  return PICKING ? 'setup' : DRAFT_MODE;
}

// The board column: the tabs, the board itself, the tray or the teach panel under it,
// and *and now try* at the foot of it. Drawn from `drawVerdict` as well as on its own,
// so every existing redraw reaches it.
function drawBoardSide() {
  const side = document.getElementById('boardSide');
  if (!side) return;
  const shown = benchShown();
  // A game in hand owns this column, and every check comes through here -- so rebuilding
  // the column would throw the game away between two of its own moves. The board keeps
  // itself current; only the panel under it is refreshed.
  if (BENCH_PLAY) {
    if (shown === 'play') return drawTry();
    endBenchGame('The board went back to the editor, so the game you had in hand is over.');
  }
  const preview = lastPreview();
  // A hex board takes the larger share of a computer's three columns (Q545, `style.py`).
  const bench = side.closest ? side.closest('.designer') : null;
  if (bench) bench.classList.toggle('hexBench', !!(preview && preview.lattice === 'hex'));
  const invalid = !!(DRAFT_CHECK && !DRAFT_CHECK.valid);
  // An invalid draft's own board, drawn from a stand-in (`designing.board_preview`), is
  // this draft's board and army; only a board kept from an earlier check is stale.
  const own = invalid && !!(DRAFT_CHECK.preview && DRAFT_CHECK.preview.standIn);
  const stale = invalid && !own;
  // The keyboard's place on the board survives the column being drawn again (D9): every
  // action on the board redraws it, and a person working it by keys keeps working.
  const keyed = !!(document.activeElement && document.activeElement.closest
    && document.activeElement.closest('#boardSide figure.benchBoard'));
  side.innerHTML = `<div class="panel benchPanel">
    <div class="row small benchTabs">${benchTabsHtml()}</div>
    ${shown === 'play' ? `<div id="benchPlay">${benchPlayHtml()}</div>`
      : preview ? `<div class="benchRow">
        <div class="tray"${shown === 'setup' ? ' data-palette' : ''}>${
          shown === 'setup' ? paletteHtml() : pieceTrayHtml()}</div>
        <figure data-board="${shown === 'teach' ? 'alone' : 'start'}" data-frame="board"
          tabindex="0" aria-label="the board: arrow keys move from square to square, Enter acts
          on the square, Esc cancels" aria-describedby="benchSaid"
          class="benchBoard${stale ? ' stale' : ''}">${benchBoardHtml(preview, shown)}</figure>
      </div>
      <div data-saidslot></div>
      <div class="small muted benchHover" id="benchHover" aria-live="polite"></div>
      ${stale ? `<div class="muted small u-mt-6">That is the last
        position these rules made, not this draft. Putting a piece down and picking
        squares still work on it, and it redraws as soon as the draft is a game
        again.</div>` : ''}
      ${own ? `<div class="muted small u-mt-6">This draft's board and
        where its pieces stand. The rules are not a game yet, so nothing moves on it until
        they are; putting a piece down and picking squares still work.</div>` : ''}
      ${benchSaysHtml(shown)}`
      : `<div class="muted small">The board appears as soon as these rules make a
         game. What is wrong with them is beside this.</div>`}
  </div>${tryPanelHtml()}`;
  side.querySelectorAll('[data-bench]').forEach(node => {
    node.onclick = () => benchMode(node.dataset.bench);
  });
  placeBenchSaid(side);
  wireWildLinks(side);
  wirePalette(side);
  wirePortalRow(side);
  wireWildRow(side);
  wirePieceTray(side);
  wireBoards(side);
  wireBenchPlay();
  wireTry();
  if (keyed) {
    const figure = side.querySelector('figure.benchBoard');
    if (figure && typeof figure.focus === 'function') figure.focus();
  }
}

// The one board, drawn for whichever question is being asked of it. Dimmed while the
// draft is not a game, because then it is the position before the edit and not this
// draft -- and it still takes clicks, which is how a half-made edit is repaired.
function benchBoardHtml(preview, shown) {
  const picked = pickedSquares(preview.width);
  const bench = benchMarks(shown);
  // Each wild piece's walk is drawn on the set-up board, always (Q852), and on no other:
  // *where it goes* is one piece's answer, and the walks would cross its dots.
  if (shown !== 'setup') preview = Object.assign({}, preview, { walks: undefined });
  // The walk being picked is drawn as the draft holds it, before the check is back (Q854).
  else preview = pickedWalk(preview);
  if (shown !== 'teach') {
    return boardPreview(preview, shown === 'goes' ? liveMap() : null, picked, bench);
  }
  const map = DRAFT_MAP && !DRAFT_MAP.live ? DRAFT_MAP : null;
  if (!map) return boardPreview(preview, null, picked, bench);
  // While a move is being taught the marks are the chosen reading's, which is the
  // engine's answer about that descriptor, and the squares that were clicked are tinted
  // under them -- so a reading that reaches more than was asked for shows it.
  const teaching = TEACHING;
  const offer = teaching ? teachingOffer() : null;
  const answer = teaching ? (offer || { moves: [], takes: [] }) : map;
  const from = teaching ? teaching.square : map.from;
  const marks = { from, moves: answer.moves || [], takes: answer.takes || [] };
  // The engine's transits for the piece on its own among the portals (N5, D5), while the
  // map is the descriptors' own and no reading is being tried instead.
  if (!teaching && map.transits) Object.assign(marks, { transits: map.transits, direct: map.direct });
  const clicked = teaching ? { moves: teaching.targets, takes: [] } : picked;
  return boardPreview(loneBoard(preview, from, teaching ? teaching.value : map.value),
                      marks, clicked, bench);
}

// The live answer, and only when it is one: a mode switch redraws before the fetch it
// started has come back, and drawing an empty-board answer on the opening position
// would put a bishop's whole star through its own pawns.
function liveMap() {
  return DRAFT_MAP && DRAFT_MAP.live ? DRAFT_MAP : null;
}

// What the board under it is showing, in one sentence, plus the controls that mode has.
function benchSaysHtml(shown) {
  const legend = benchLegendHtml(shown);
  // While squares are picked the note that says so comes first, under the board, with its
  // *a whole row* and *done picking*; the tray's note is for the tray, which picking has
  // put away (Q854).
  if (shown === 'setup' && PICKING) {
    return pickNoteHtml() + legend + walkLeapsHtml(pickedWalk(lastPreview()), true);
  }
  if (shown === 'setup' && WILD_TOOL) return wildToolHtml() + legend + walkLeapsHtml(lastPreview(), true);
  if (shown === 'setup') return legend + walkLeapsHtml(lastPreview(), true) + paletteNoteHtml();
  const map = DRAFT_MAP;
  const piece = (DRAFT.pieces || [])[DRAFT_PIECE] || {};
  // Where a piece can come out of one end two ways round, it is said once (D6); and the
  // key to momentum, once a sitting, wherever the board has portals (D5).
  const choice = shown === 'goes' && map && map.live ? choiceLineHtml(map, piece.name || piece.id) : '';
  return `${legend}${TEACHING ? '' : moveKeyHtml()}${choice}${portalKeyHtml()}${pickNoteHtml()}${
    shown === 'teach' ? teachPanelHtml() : ''}`;
}

// The legend under the bench board, for the marks it draws beyond the engine's dots: the
// rules' marks on squares, the walls and the portals, each named from its `MARK_KINDS` row
// as the play board names it (S3).
function benchLegendHtml(shown) {
  const preview = lastPreview();
  if (!preview || typeof legendEntries !== 'function') return '';
  const kinds = new Set(boardKinds(preview.obstacles, preview.portals));
  // The wild pieces' walks, where the set-up board draws them (Q852).
  if (shown === 'setup' && walksDrawn(preview)) kinds.add('walk');
  const marks = squareMarks(preview, { zone: !zonesShown() });
  Object.keys(marks).forEach(square => marks[square].forEach(mark => kinds.add(mark.kind)));
  const said = legendEntries(kinds, preview.wrap, '', preview.portals);
  return said.length ? `<div class="frame-legend small benchLegend">on the board: ${said.join(' ')}</div>` : '';
}

// ------------------------------------------------------------- playing it here
//
// Option B4, the last of the workbench stages. Trying a change used to mean leaving the
// editor for the board page and coming back, which is two page draws for one look
// (finding 2.4) -- so the fourth tab plays the draft on the board already in front of
// you, against the machine or by hand for both sides.
//
// **It is the same board the play page draws**, mounted in this column instead of in the
// whole view: `showBoard` takes a mount, `page/board.js` answers every question about the
// position, and nothing here works out a legal move, an outcome or the machine's reply.
// The draft travels in the request body to `POST /api/design/play`, which is the route
// *play it on its own page* has always used, so nothing is written and no game played
// here is recorded as a rated or a human one.
//
// **A game in hand belongs to the rules it started under.** Editing the draft while one
// is running would leave a board playing a game nobody can see the rules of, so the first
// change ends it and says so.

let BENCH_PLAY = null,   // { key, id } -- the draft a game in hand was started from
    BENCH_ENDED = '';    // why the last one stopped, for as long as that sentence holds

// The invitation, which is what the tab shows whenever no game is in hand. Once one is,
// the board writes over this whole element and keeps it until the game ends.
function benchPlayHtml() {
  const said = BENCH_ENDED
    ? `<div class="ok small u-m-0-0-8">${esc(BENCH_ENDED)}</div>` : '';
  if (!DRAFT_CHECK || !DRAFT_CHECK.valid) {
    return `${said}<div class="muted small">A draft can be played as soon as these rules
      make a game. What is wrong with them is beside this.</div>`;
  }
  return `${said}
    <div class="muted small">A game of the rules as they stand right now, played here.
      Nothing is kept, nothing is rated, and the first change you make to the draft ends
      the game. There is no move list on this board: <i>${esc(PLAY_AWAY)}</i> opens the
      full board with one.</div>
    <div class="row u-mt-8">
      <button class="primary" data-seats="machine">you against the machine</button>
      <button data-seats="hand">both sides by hand</button>
      <button data-seats="watch" title="${esc(watchTitle(watchesRounds()))}">watch it play itself</button>
    </div>`;
}

// Both are rungs of the rated ladder, the same two the verdict column seats.
function benchSeats(which) {
  if (which === 'hand') return ['human', 'human'];
  if (which === 'watch') return [WATCH_AGENT, WATCH_AGENT];
  return ['human', PLAY_AGENT];
}

function wireBenchPlay() {
  document.querySelectorAll('[data-seats]').forEach(node => {
    node.onclick = () => playDraftHere(benchSeats(node.dataset.seats));
  });
}

// The spinner goes in the board's own panel rather than at the top of the page: on a
// phone the top of the page is several screens away from the button that was pressed.
function playDraftHere(seats) {
  const box = document.getElementById('benchPlay');
  if (box) {
    box.innerHTML = '<div class="busy"><span class="spin"></span> Starting the game…</div>';
  }
  return playDraft(seats, undefined, true);
}

// The snapshot, put on the board this column keeps. `showBoard` does the rest, exactly
// as it does for a game opened from the library.
function mountBenchGame(snapshot) {
  const mount = document.getElementById('benchPlay');
  if (!mount) return;                       // the editor moved on while it was starting
  BENCH_PLAY = { key: JSON.stringify(DRAFT), id: snapshot.id };
  BENCH_ENDED = '';
  return showBoard(snapshot, { mount });
}

// The game given up, and the board handed back to the editor.
function stopPlayingHere() {
  endBenchGame('');
  drawBoardSide();
}

// A game in hand is over the moment the draft moves. `why` is the sentence that stands
// in its place; somebody who pressed *back to editing* or another tab knows what they
// pressed and gets none.
function endBenchGame(why) {
  if (!BENCH_PLAY) return;
  BENCH_PLAY = null;
  BENCH_ENDED = why || '';
  leaveBoard();
  // `showBoard` turns the board round for whoever sits at the far end. The editor's own
  // boards are always drawn with the first player at the bottom.
  FLIPPED = false;
}

// Redrawn whole on every structural change -- a piece added, a move removed -- and never
// on a keystroke. Text inputs write straight into DRAFT and ask for a re-check; redrawing
// under a cursor would put it back at the start of the field on every letter.
function drawDesigner() {
  const form = document.getElementById('form');
  if (!form) return;
  const board = DRAFT.board || {};
  const rules = DRAFT.rules || {};
  // A picker open in the card of a piece that has since gone is closed.
  if (typeof PICK_AT === 'number' && !(DRAFT.pieces || [])[PICK_AT]) PICK_AT = null;

  form.innerHTML = `
    <div class="panel">
      <div class="row small u-mb-10">
        <button id="dUndo" title="Ctrl+Z">undo</button>
        <button id="dRedo" title="Ctrl+Y">redo</button>
        <span class="muted">Every change can be taken back, with these or with Ctrl+Z and
          Ctrl+Y, and the draft survives a reload of this tab.</span>
      </div>
      <label class="field"><span>what it is called</span>
        <input id="dName" value="${esc(DRAFT.name || '')}"></label>
      <label class="field"><span>a sentence about it, for the library</span>
        <textarea id="dDesc" class="u-minh-52"
          placeholder="What is this game? One sentence, in your own words: the library shows it under the name."
          >${esc(DRAFT.description || '')}</textarea></label>
      ${boardFormHtml(board)}
      <div id="dTurns">${turnsHtml()}</div>
    </div>

    <h3>The pieces</h3>
    <div class="editor">${piecesInOrder().map(index => pieceEditor(DRAFT.pieces[index], index)).join('')}</div>
    <div class="row u-mt-10">
      <button id="dAddPiece">add a piece</button>
      <span id="dRestAllBox">${restAllHtml()}</span>
      <span class="muted small">Click a piece to see where it can go.</span>
    </div>
    <div id="dRestSaid" aria-live="polite">${restSaidHtml()}</div>
    ${pickOpenerHtml()}

    <h3>Where they start</h3>
    <div class="panel">
      <div class="muted small u-m-0-0-8">
        The pieces go down on the big board: pick one from the tray beside it and click.
        These rows are the same thing written out, for reading and for typing a whole
        rank at once.
      </div>
      <div class="muted small u-m-8-0">
        Squares are named the way the board labels them, so <code>a1</code> is the bottom
        left. ${(DRAFT.setup || {}).mirror !== false
          ? 'The second player gets the same army, so every piece you put down is put'
            + ' down for both.'
          : 'Each player is placed separately, and each row says whose it is.'}
      </div>
      ${((DRAFT.setup || {}).placements || []).map(placementEditor).join('')}
      <div class="row u-mt-8">
        <button id="dAddPlacement">place a piece</button>
        <label class="pick"><input type="checkbox" id="dMirror"${
          (DRAFT.setup || {}).mirror !== false ? ' checked' : ''}> mirror for the other player</label>
      </div>
      <label class="field u-mt-8"><span>the second player's side of the board is</span>
        <select id="dOrientation">
          ${[['mirror', 'reflected across the middle, files kept, as in chess'],
             ['rotate', 'the board turned round, files reversed, as in xiangqi']]
            .map(([value, label]) => `<option value="${value}"${
              ((DRAFT.setup || {}).orientation || 'mirror') === value ? ' selected' : ''
            }>${label}</option>`).join('')}
        </select></label>
    </div>

    <h3>The rules</h3>
    <div class="panel">
      <div class="muted small u-mb-8">
        Each rule has a card, and the sentence under each control says what it does. A
        rule with no card is not in the game.
      </div>
      ${rulesCardsHtml([])}
    </div>

    <details class="panel u-mt-14">
      <summary class="muted small">${esc(pasteLabel())} (JSON), to copy out or paste in</summary>
      <textarea id="dJson" class="jsonText" spellcheck="false">${
        esc(JSON.stringify(DRAFT, null, 1))}</textarea>
      <button id="dJsonApply" class="u-mt-6">use this</button>
      <span class="muted small">Paste a game's text here to edit it. It is checked first,
        and a text with a mistake is not taken.</span>
      <div id="dPasteSaid" class="ruleSaid" aria-live="polite"></div>
    </details>`;

  wireDesigner();
  wireTurns(form);
  wireRestAll(form);
}

// ------------------------------------------------------------- every piece rests
//
// Q995d (`docs/reference/designer-coverage.md` fix 7). Schema 29 made a rest each piece's
// own, in a game by turns or in rounds, and every game that rests rests every piece the
// same: setting that was a field on each card, one card at a time. One control beside *add
// a piece* sets every piece at once, and each card's field (*how it takes its turns*) stays
// for the exceptions. The rule language has no rest for the whole game, so the choice is
// written out on each piece, by the server, which asks the validator what may take one
// (`designing.rest_every_piece`); the page never decides. The words are `spec.plain`'s
// (`REST_WORDS`, sent in the card model). Under the control, a sentence says what every
// piece rests, read off the draft as it stands.

let REST_SAID = null,     // { key, rest, changes } after a choice; { key, rest, errors } for a refused one
    REST_ASKING = false;  // while the server answers

function restWords() {
  return (cardModel() || {}).restWords || {};
}

// `{name}` in a word template, filled in.
function fillWords(template, values) {
  return String(template || '').replace(/\{(\w+)\}/g,
    (whole, key) => (values[key] === undefined ? whole : values[key]));
}

// A piece's rest as written: absent is 0, it never rests.
function restOf(piece) {
  const rest = (piece || {}).rest;
  return Number.isInteger(rest) && rest > 0 ? rest : 0;
}

// *1 turn*, *2 rounds*: in a game played in rounds a round is a turn.
function restCount(rest) {
  const units = (restWords().units || {})[inRounds() ? 'rounds' : 'turns'] || ['turn', 'turns'];
  return `${rest} ${rest === 1 ? units[0] : units[1]}`;
}

// What a card's head says beside the piece's name: *rests 2 turns*, or nothing.
function pieceRestWords(piece) {
  const rest = restOf(piece);
  return rest ? fillWords(restWords().head, { rest: restCount(rest) }) : '';
}

// The longest rest the menu offers, the schema's (`pieces[].rest`'s maximum).
function restMost() {
  const field = ((cardModel() || {}).piece || []).find(f => f.key === 'rest');
  return field && Number.isInteger(field.maximum) ? field.maximum : 0;
}

// The rest every piece shares, 0 for none, or null while they differ.
function restShared() {
  const rests = new Set((DRAFT.pieces || []).map(restOf));
  return rests.size === 1 ? [...rests][0] : (rests.size ? null : 0);
}

function restNames(names) {
  if (names.length < 2) return names.join('');
  return names.slice(0, -1).join(', ') + ` ${restWords().and || 'and'} ` + names[names.length - 1];
}

// What every piece rests, in one sentence: the same for all, or each group by its rest,
// the largest group first and the pieces that never rest last.
function restSentence() {
  const words = restWords();
  const pieces = DRAFT.pieces || [];
  if (!pieces.length || !words.label) return '';
  const shared = restShared();
  if (shared === 0) return (words.none || {})[inRounds() ? 'rounds' : 'turns'] || '';
  if (shared) return fillWords(words.every, { rest: restCount(shared) });
  const groups = new Map();
  pieces.forEach((piece, index) => {
    const rest = restOf(piece);
    if (!groups.has(rest)) groups.set(rest, []);
    groups.get(rest).push((piece || {}).name || (piece || {}).id || `piece ${index + 1}`);
  });
  const order = [...groups.keys()].sort((a, b) => (a === 0) - (b === 0)
    || groups.get(b).length - groups.get(a).length || a - b);
  const parts = order.map(rest => (rest
    ? fillWords(words.group, { rest: restCount(rest), names: restNames(groups.get(rest)) })
    : fillWords(words.groupNever, { names: restNames(groups.get(rest)) })));
  return fillWords(words.each, { groups: parts.join('; ') });
}

// The control: *every piece rests [1 turn] after it moves*. While the pieces differ the
// menu reads *each its own*, which is a line, not a choice.
function restAllHtml() {
  const words = restWords();
  const most = restMost();
  if (!words.label || !most) return '';
  const shared = restShared();
  const options = [];
  if (shared === null) {
    options.push(`<option value="" selected disabled>${esc(words.differ)}</option>`);
  }
  options.push(`<option value="0"${shared === 0 ? ' selected' : ''}>${esc(words.never)}</option>`);
  for (let rest = 1; rest <= most; rest++) {
    options.push(`<option value="${rest}"${shared === rest ? ' selected' : ''}>${
      esc(restCount(rest))}</option>`);
  }
  return `<label class="restAll">${esc(words.label)}
      <select id="dRestAll" data-restall="1"${REST_ASKING ? ' disabled' : ''}>${options.join('')}</select>
      ${esc(words.after)}</label>`;
}

// What the last choice's sentence belongs to: the draft as it stands.
function restKey() {
  return JSON.stringify(DRAFT);
}

// Under the control: the sentence, and what the last choice did.
function restSaidHtml() {
  const words = restWords();
  if (!words.label) return '';
  const said = REST_SAID && REST_SAID.key === restKey() ? REST_SAID : null;
  const sentence = restSentence();
  const out = [];
  if (sentence) {
    out.push(`<div class="muted small restSentence">${esc(sentence)}${
      restShared() === null ? '' : ` ${esc(words.override)}`}</div>`);
  }
  if (said && said.errors && said.errors.length) {
    const choice = `${words.label} ${said.rest ? restCount(said.rest) : words.never} ${words.after}`;
    out.push(`<div class="error small turnSaid"><b>${esc(fillWords(words.refused, { choice }))}</b>
      <ul>${said.errors.map(line => `<li>${esc(line)}</li>`).join('')}</ul></div>`);
  } else if (said) {
    const changes = said.changes || [];
    out.push(`<div class="ok small turnSaid">${changes.length === 1 ? esc(changes[0])
      : `<ul>${changes.map(line => `<li>${esc(line)}</li>`).join('')}</ul>`}
      <button class="quiet" id="dRestUndo">undo this</button></div>`);
  }
  return out.join('');
}

// Redrawn where a card's own rest is typed, and when a check answers, without the form.
function drawRestAll() {
  const box = document.getElementById('dRestAllBox');
  const said = document.getElementById('dRestSaid');
  if (box) box.innerHTML = restAllHtml();
  if (said) said.innerHTML = restSaidHtml();
  wireRestAll(document);
}

function wireRestAll(root) {
  if (!root || !root.querySelector) return;
  const menu = root.querySelector('#dRestAll');
  if (menu) menu.onchange = () => { if (menu.value !== '') chooseRestAll(Number(menu.value)); };
  const undo = root.querySelector('#dRestUndo');
  if (undo) undo.onclick = () => stepDraft(false);
}

// One press, as *how the players take turns* makes one: the draft as the server writes it,
// recorded at once so that undo is the step before it, and drawn with what changed.
function chooseRestAll(rest) {
  if (REST_ASKING) return;
  const from = JSON.stringify(DRAFT);
  const key = restKey();
  REST_ASKING = true;
  drawRestAll();
  return api('/api/design/rest', { spec: DRAFT, rest }).then(answer => {
    REST_ASKING = false;
    if (JSON.stringify(DRAFT) !== from) { drawRestAll(); return; }
    if (!answer || !answer.draft) {
      REST_SAID = { key, rest, errors: (answer && answer.errors) || [] };
      drawRestAll();
      return;
    }
    snapshotDraft();
    DRAFT = answer.draft;
    snapshotDraft();
    REST_SAID = { key: restKey(), rest, changes: answer.changes || [] };
    drawDesigner();
    checkDraft();
    loadMoveMap();
  }).catch(error => {
    REST_ASKING = false;
    REST_SAID = { key, rest, errors: [errorWords(error)] };
    drawRestAll();
  });
}

// ------------------------------------------------------- how the players take turns
//
// Q995b (`docs/reference/designer-coverage.md` F1, F2). A game in rounds was a rule card
// at the foot of the rules panel under a name without *round*, and from a copy of chess it
// was not in the menu at all: three conflicts stood in the way, each named alone, in the
// validator's voice. Now the way the players take turns is the first panel's, beside the
// board's size, and choosing one asks the server for the draft played that way. The
// server makes every change the way needs and says each (`designing.take_turns`); which
// rules are in the way is the validator's, never this page's. The whole step is one
// draft replaced, so one undo takes it back.

const TURN_WAYS = [
  ['alternate', 'one after the other', 'a player moves, then the other, as in chess'],
  ['rounds', 'in rounds', 'both players choose a move in secret, and both moves are made at once'],
  ['several', 'more than one move a turn', 'each player makes several moves in a row'],
];
// What a choice is called in the sentence that follows it.
const TURN_NOW = { alternate: 'one move after the other', rounds: 'in rounds',
                   several: 'with more than one move a turn' };

let TURN_SAID = null,    // { key, way, changes } after a choice; { key, way, errors } for a refused one
    TURN_ASKING = '';    // the way being asked for, while the server answers

// Which way the draft is played now, read off the rules it states.
function turnWay() {
  const rules = DRAFT.rules || {};
  if (rules.simultaneous && typeof rules.simultaneous === 'object') return 'rounds';
  if (rules.turn && typeof rules.turn === 'object') return 'several';
  return 'alternate';
}

function turnsHtml() {
  const now = TURN_ASKING || turnWay();
  const said = TURN_SAID && TURN_SAID.key === JSON.stringify(DRAFT) ? TURN_SAID : null;
  return `<fieldset class="turnWays">
      <legend>how the players take turns</legend>
      ${TURN_WAYS.map(([way, label, says]) => `<label class="turnWay">
        <input type="radio" name="dTurnWay" value="${way}" data-turnway="${way}"${
          way === now ? ' checked' : ''}${TURN_ASKING ? ' disabled' : ''}>
        <span><b>${esc(label)}</b><span class="muted small cardSays">${esc(says)}</span></span>
      </label>`).join('')}
      <div id="dTurnSaid" aria-live="polite">${turnSaidHtml(said)}</div>
    </fieldset>`;
}

// What the last choice changed, or why it could not be made: every reason at once.
function turnSaidHtml(said) {
  if (!said) return '';
  const label = (TURN_WAYS.find(([way]) => way === said.way) || [])[1] || said.way;
  if (said.errors && said.errors.length) {
    return `<div class="error small turnSaid"><b>'${esc(label)}' can't be chosen for this game
      yet</b>, so nothing was changed. ${said.errors.length === 1 ? 'The reason'
        : `All ${said.errors.length} reasons`}:
      <ul>${said.errors.map(line => `<li>${esc(line)}</li>`).join('')}</ul></div>`;
  }
  const changes = said.changes || [];
  return `<div class="ok small turnSaid">Now played ${esc(TURN_NOW[said.way] || label)}.
      ${changes.length ? `For that, ${changes.length === 1 ? 'this' : `these ${changes.length}`}
        changed too:<ul>${changes.map(line => `<li>${esc(line)}</li>`).join('')}</ul>`
        : 'Nothing else needed to change.'}
      <button class="quiet" id="dTurnUndo">undo this</button></div>`;
}

function drawTurns() {
  const box = document.getElementById('dTurns');
  if (!box) return;
  box.innerHTML = turnsHtml();
  wireTurns(box);
}

function wireTurns(root) {
  if (!root || !root.querySelectorAll) return;
  root.querySelectorAll('[data-turnway]').forEach(node => {
    node.onchange = () => { if (node.checked) chooseTurnWay(node.dataset.turnway); };
  });
  const undo = root.querySelector('#dTurnUndo');
  if (undo) undo.onclick = () => stepDraft(false);
}

// One press: the draft as the server makes it for this way, recorded at once so that undo
// is the step before it, and drawn with what changed.
function chooseTurnWay(way) {
  if (TURN_ASKING || way === turnWay()) return;
  const model = cardModel();
  const key = { rounds: 'simultaneous', several: 'turn' }[way];
  const field = key && model ? model.rules.find(f => f.key === key) : null;
  const card = field ? fieldDefault(field, [key], 'rules') : null;
  const from = JSON.stringify(DRAFT);
  TURN_ASKING = way;
  drawTurns();
  return api('/api/design/turns', { spec: DRAFT, way, card }).then(answer => {
    TURN_ASKING = '';
    if (JSON.stringify(DRAFT) !== from) { drawTurns(); return; }
    if (!answer || !answer.draft) {
      TURN_SAID = { key: from, way, errors: (answer && answer.errors) || [] };
      drawTurns();
      return;
    }
    snapshotDraft();
    DRAFT = answer.draft;
    snapshotDraft();
    TURN_SAID = { key: JSON.stringify(DRAFT), way, changes: answer.changes || [] };
    drawDesigner();
    checkDraft();
    loadMoveMap();
  }).catch(error => {
    TURN_ASKING = '';
    TURN_SAID = { key: from, way, errors: [errorWords(error)] };
    drawTurns();
  });
}

// ------------------------------------------------ a piece out of a game that already plays
//
// Option D1. Every distinct piece the library holds, said in the sentences its own game's
// page says it in, and taken whole rather than retyped. Three of the four walks entered
// the same king's step two or three times from a new piece's default (finding 2.6).
//
// The rows are the server's (`designing.piece_palette`), which reads them off the specs.
// Nothing here knows a piece: this filters by a typed word, gives the new piece a letter
// the draft has free, and adds it.

// The pieces on offer: those written for the lattice this draft is on, because a vector
// on a hex board points somewhere else, narrowed by whatever is typed into the picker.
function paletteOffers() {
  const all = (DESIGN_KIT && DESIGN_KIT.palette) || [];
  const lattice = (DRAFT.board || {}).lattice || 'square';
  const want = PICK_FIND.trim().toLowerCase();
  return all.filter(entry => entry.lattice === lattice && (!want
    || (entry.name + ' ' + entry.says.join(' ') + ' ' + entry.games.join(' '))
       .toLowerCase().includes(want)));
}

function paletteListHtml() {
  const offers = paletteOffers();
  if (!offers.length) return '<div class="muted small">Nothing in the library says that.</div>';
  return `<ul class="paletteList">${offers.map(entry => `<li>
      <button data-palettepiece="${esc(entry.key)}">${esc(entry.name)}</button>
      <span class="muted small">${entry.says.map(esc).join('; ')}</span>
      <span class="muted small paletteGames" title="${esc(entry.games.join(', '))}">in ${
        entry.games.length} game${entry.games.length === 1 ? '' : 's'}</span>
    </li>`).join('')}</ul>`;
}

// The letter a new piece gets: the one the library calls it, when this draft has it free.
function freeLetter(wanted) {
  const used = new Set((DRAFT.pieces || []).map(p => p.id));
  if (wanted && !used.has(wanted)) return wanted;
  return 'ABCDEFGHJLMTUVWXYZ'.split('').find(c => !used.has(c)) || 'Z';
}

// A piece just added, by its letter, for as long as no square holds it (Q532). A piece
// taken from the palette used to arrive in the list of pieces and nowhere on the board,
// with nothing saying so, and a person who then pressed *play it here* met a game without
// it. Only the piece just added is said to be missing: a piece that only arrives by
// promotion or from a hand stands nowhere at the start either, and is meant to.
let FRESH_PIECE = null;

// Whether any placement row puts the piece with this letter on a square. Read off the
// draft this page is editing, which is what the rows under *where they start* show.
function standsSomewhere(id) {
  return ((DRAFT.setup || {}).placements || [])
    .some(row => row.piece === id && (row.squares || []).length);
}

// The piece just added, when it is still off the board: its index, else -1.
function freshUnplaced() {
  if (FRESH_PIECE == null || standsSomewhere(FRESH_PIECE)) return -1;
  return (DRAFT.pieces || []).findIndex(piece => piece.id === FRESH_PIECE);
}

// Said in the new piece's own card and under the board, the two places a person is
// looking after pressing to add one.
function freshNoteHtml(index) {
  if (index < 0 || index !== freshUnplaced()) return '';
  const piece = DRAFT.pieces[index];
  return `<div class="freshNote small" data-fresh="${index}">The ${esc(piece.name || piece.id)}
    is not on the board yet. It is picked in the tray: click a square on the board to put
    it down. A piece that only arrives by promotion or from a hand can stay off it.
    </div>`;
}

// A new piece, made ready to put down: the board goes to *set up* with it picked in the
// tray, so the next click on a square places it.
function freshPiece(index) {
  FRESH_PIECE = DRAFT.pieces[index].id;
  PIECES_OPEN.add(FRESH_PIECE);
  PLACING = { piece: index, player: 0 };
  if (!PICKING && DRAFT_MODE !== 'setup') {
    DRAFT_MODE = 'setup';
    TEACHING = null;
    endBenchGame('');
  }
}

function takePalettePiece(key) {
  const entry = ((DESIGN_KIT && DESIGN_KIT.palette) || []).find(one => one.key === key);
  if (!entry) return;
  DRAFT.pieces = DRAFT.pieces || [];
  DRAFT.pieces.push(Object.assign(JSON.parse(JSON.stringify(entry.piece)),
                                  { id: freeLetter(entry.id) }));
  DRAFT_PIECE = DRAFT.pieces.length - 1;
  DRAFT_SQUARE = -1;
  freshPiece(DRAFT_PIECE);
  drawDesigner();
  checkDraft();
  loadMoveMap();
  showNewPiece(DRAFT_PIECE);
}

// ------------------------------------------------ the one piece picker
//
// PF4. One place to choose a piece out of the library (the user, 2026-10-26: the drawing
// menu under each piece, the ready-made panel and the library games' panel were three
// places). Its tiles are the drawings (`web/piece_library.py`, `GET /api/design?part=pieces`,
// fetched the first time it opens, since they are a quarter of a megabyte of pictures). A
// drawing whose movement tradition settles comes with its ready piece
// (`assets/pieces/ready.json`): taken, it brings the drawing, the name and the moves, all the
// draft's own after. A drawing without one gives just the drawing. Under the tiles, the
// pieces of the library's games (`designing.piece_palette`), taken whole by their moves.
//
// The picker stands in one place at a time: under the list of pieces, for a new piece, or
// in a piece's card, opened by its *choose a drawing*, where a drawing goes on that piece.
// A press on a tile does not take it: it opens a line under the tile's row -- the drawing,
// the description, the moves in words and the buttons that take it -- so what a piece is
// is read by tap, click or Enter before anything changes, and Escape closes it. (PF3's
// small i in a tile's corner was missed on a phone, where a tap on the tile itself took
// the piece at once.) Nothing here knows a piece: a rule of the whole game a ready piece
// needs is the entry's own data, written only when asked, '@self' its letter and '@royals'
// the draft's royal pieces.

let PIECE_LIBRARY = null;   // {categories, pieces}, once fetched
let PICK_AT = null;         // where the picker is open: 'new', a piece's index, or null
let PICK_FIND = '';         // what is typed in its search
let PICK_KIND = '';         // the category whose chip is on, or '' for all of them
let PICK_MOVING = true;     // for a new piece, only the drawings that come with moves
let PICK_SHOWN = '';        // the drawing whose line is open under its row, or ''
const READY_NEEDS = {};     // a ready piece's letter -> what it said it needs, until dismissed

// A drawing's ready piece, when it is written for the lattice this draft is on.
function readyHere(drawing) {
  const ready = drawing && drawing.ready;
  const lattice = (DRAFT.board || {}).lattice || 'square';
  return ready && ready.lattice === lattice ? ready : null;
}

// The piece the picker dresses, when it is open in a piece's card; else null.
function pickTarget(at) {
  return typeof at === 'number' ? (DRAFT.pieces || [])[at] || null : null;
}

function pickOpenerHtml() {
  const open = PICK_AT === 'new';
  return `<div class="row u-mt-10">
      <button id="dPickNew" aria-expanded="${open}" aria-controls="dPicker">${
        open ? 'close the piece library' : 'choose from the piece library'}</button>
      <span class="muted small">a drawing with its moves, a drawing alone, or a piece out of
        a game</span>
    </div>
    <div data-pickat="new">${open ? pickerHtml('new') : ''}</div>`;
}

function pickerHtml(at) {
  const piece = pickTarget(at);
  const hexy = ((DRAFT.board || {}).lattice || 'square') === 'hex';
  const says = piece
    ? `Press a drawing to see it, then put it on the ${esc(piece.name || piece.id)}. A
       drawing that comes with moves can be added as a piece of its own instead.`
    : `Press a piece to read what it is and how it moves, then add it. One that comes with
       moves brings its drawing, its name and its moves, all yours to change after; one that
       needs a rule of the whole game says so on its card.${hexy
         ? ' Only moves written for a hex board are offered.' : ''}`;
  return `<div class="panel picker u-mt-8" id="dPicker">
    <div class="muted small u-m-0-0-8">${says}</div>
    <label class="field"><span>find one</span>
      <input id="dPickFind" type="search" value="${esc(PICK_FIND)}" autocomplete="off"
        placeholder="a name or a way of moving: camel, rider, zigzag"></label>
    <div class="chips" role="group" aria-label="kinds of piece" id="dPickKinds">${
      pickKindsHtml(at)}</div>
    ${piece ? '' : `<label class="pick small"><input type="checkbox" id="dPickMoving"${
      PICK_MOVING ? ' checked' : ''}> only the drawings that come with moves</label>`}
    <div class="muted small u-mt-6" id="dPickCount" aria-live="polite">${
      esc(pickCountWords(at))}</div>
    <div id="dPickList">${pickListHtml(at)}</div>
  </div>`;
}

// The drawings on offer: the chip's category (unless `anyKind`), narrowed by every word
// typed, each matched anywhere in what the drawing and its ready piece are called and say.
// For a new piece, while the box is ticked, only the drawings that come with moves here.
function pickOffers(at, anyKind) {
  if (!PIECE_LIBRARY) return [];
  const words = PICK_FIND.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const moving = at === 'new' && PICK_MOVING;
  return PIECE_LIBRARY.pieces.filter(drawing => {
    const ready = readyHere(drawing);
    if (moving && !ready) return false;
    if (!anyKind && PICK_KIND && drawing.category !== PICK_KIND) return false;
    const said = [drawing.name, drawing.slug, drawing.says, drawing.category]
      .concat(drawing.words, ready ? [ready.piece.name, ready.reading, ready.description]
        .concat(ready.says) : []).join(' ').toLowerCase();
    return words.every(word => said.includes(word));
  });
}

function pickKindsHtml(at) {
  if (!PIECE_LIBRARY) return '';
  const used = new Set(pickOffers(at, true).map(drawing => drawing.category));
  return [{ key: '', label: 'all' }].concat(PIECE_LIBRARY.categories
    .filter(kind => used.has(kind.key))).map(kind => {
    const on = PICK_KIND === kind.key;
    return `<button class="chip${on ? ' on' : ''}" data-pickkind="${esc(kind.key)}"
      aria-pressed="${on}">${esc(kind.label)}</button>`;
  }).join('');
}

function pickCountWords(at) {
  if (!PIECE_LIBRARY) return '';
  const noun = at === 'new' && PICK_MOVING ? 'pieces with moves' : 'drawings';
  return `${pickOffers(at, false).length} of ${pickOffers(at, true).length} ${noun}; `
    + 'press one to read about it.';
}

function pickListHtml(at) {
  if (!PIECE_LIBRARY) return '<div class="muted small">Fetching the pieces...</div>';
  const offers = pickOffers(at, false);
  const piece = pickTarget(at);
  const worn = piece ? iconNameOf(piece.id) : '';
  // Where every tile comes with moves, no tile needs to say so.
  const tagged = !(at === 'new' && PICK_MOVING);
  const tiles = offers.length ? `<ul class="drawingGrid" id="dPickGrid">${offers.map(drawing => {
    const ready = readyHere(drawing);
    const name = ready ? ready.piece.name : drawing.name;
    const about = ready ? ready.description || ready.reading : drawing.says;
    const wearing = worn === drawing.slug;
    return `<li><button data-pick="${esc(drawing.slug)}" aria-expanded="${
      PICK_SHOWN === drawing.slug}"${wearing ? ' class="worn"' : ''}
      title="${esc(name + (about ? ': ' + about : ''))}"><img
      src="${esc(drawingHref(drawing))}" alt=""><span>${esc(name)}</span>${
      ready && tagged ? '<span class="tileTag">with moves</span>' : ''}${
      wearing ? '<span class="tileTag">worn now</span>' : ''}</button></li>`;
  }).join('')}</ul>` : '<div class="muted small">No drawing is called that.</div>';
  // The library's games' pieces come with every chip but a category's.
  const games = at === 'new' && !PICK_KIND ? paletteOffers() : [];
  return tiles + (games.length ? `<div class="pickHead">Out of a game in the library</div>
    <div class="muted small">Its moves come with it, word for word. What it may capture,
      where it may stand and what it reverts to are left behind, because those name pieces
      and squares of the game it came from.</div>${paletteListHtml()}` : '');
}

// The line under a tile's row, across the whole grid: the drawing larger, what the piece
// is, how it moves in the words a game's page uses (`spec/describe.py`, served with it),
// and the buttons that take it. It pushes the rows below down rather than lying over them.
function placePickAbout() {
  const grid = document.getElementById('dPickGrid');
  const old = document.getElementById('dPickAbout');
  if (old) old.remove();
  if (!grid) return;
  const button = PICK_SHOWN && grid.querySelector(`[data-pick="${CSS.escape(PICK_SHOWN)}"]`);
  grid.querySelectorAll('[data-pick]').forEach(node => {
    node.setAttribute('aria-expanded', String(node === button));
    if (node === button) node.setAttribute('aria-controls', 'dPickAbout');
    else node.removeAttribute('aria-controls');
  });
  if (!button) { PICK_SHOWN = ''; return; }
  const drawing = PIECE_LIBRARY.pieces.find(one => one.slug === PICK_SHOWN);
  if (!drawing) { PICK_SHOWN = ''; return; }
  const ready = readyHere(drawing);
  const piece = pickTarget(PICK_AT);
  const line = document.createElement('li');
  line.className = 'pickAbout small';
  line.id = 'dPickAbout';
  line.setAttribute('role', 'region');
  line.setAttribute('aria-label', 'about the ' + (ready ? ready.piece.name : drawing.name));
  const picture = document.createElement('img');
  picture.src = drawingHref(drawing);
  picture.alt = '';
  const words = document.createElement('div');
  const said = (text, muted) => {
    const part = document.createElement('div');
    if (muted) part.className = 'muted';
    part.textContent = text;
    words.append(part);
  };
  const title = document.createElement('b');
  title.textContent = (ready ? ready.piece.name : drawing.name) + '. ';
  const first = document.createElement('div');
  first.append(title, ready ? ready.description || ready.reading
    : drawing.says || 'A drawing.');
  words.append(first);
  if (ready) {
    said('How it moves: ' + ready.says.join('; ') + '.');
    if (ready.rulesSay) said(ready.rulesSay, true);
  } else {
    said(drawing.ready
      ? 'Its moves are written for the other kind of board, so here it gives the drawing alone.'
      : 'No moves come with this drawing: it gives the picture alone.', true);
  }
  const row = document.createElement('div');
  row.className = 'row';
  const offer = (label, primary, act) => {
    const press = document.createElement('button');
    press.textContent = label;
    if (primary) press.className = 'primary';
    press.onclick = event => { event.preventDefault(); act(); };
    row.append(press);
    return press;
  };
  if (piece) {
    const wearing = iconNameOf(piece.id) === drawing.slug;
    const put = offer(wearing ? `the ${piece.name || piece.id} wears it`
      : `put it on the ${piece.name || piece.id}`, true, () => wearDrawing(PICK_AT, drawing.slug));
    put.disabled = wearing;
    if (ready) offer(`add the ${ready.piece.name} as a new piece`, false,
      () => takeReadyPiece(drawing.slug));
  } else if (ready) {
    offer(`add the ${ready.piece.name}`, true, () => takeReadyPiece(drawing.slug));
  } else {
    offer('add a piece wearing it', true, () => addPieceWearing(drawing.slug));
  }
  words.append(row);
  line.append(picture, words);
  // The last tile of the row this one stands in: the line goes after it.
  const tile = button.closest('li');
  let last = tile;
  while (last.nextElementSibling && last.nextElementSibling.offsetTop === tile.offsetTop) {
    last = last.nextElementSibling;
  }
  last.after(line);
  const shown = line.getBoundingClientRect();
  const box = grid.getBoundingClientRect();
  if (shown.bottom > box.bottom) grid.scrollTop += shown.bottom - box.bottom;
}

function togglePickAbout(slug) {
  PICK_SHOWN = PICK_SHOWN === slug ? '' : slug;
  placePickAbout();
}

function loadPieceLibrary() {
  if (PIECE_LIBRARY || loadPieceLibrary.asked) return;
  loadPieceLibrary.asked = true;
  api('/api/design?part=pieces').then(answer => {
    PIECE_LIBRARY = answer.pieceLibrary || { categories: [], pieces: [] };
    redrawPicker();
  }).catch(error => {
    loadPieceLibrary.asked = false;
    const list = document.getElementById('dPickList');
    if (list) list.textContent = 'The pieces could not be fetched: ' + error.message;
  });
}

function openPicker(at) {
  PICK_AT = PICK_AT === at ? null : at;
  PICK_SHOWN = '';
  redrawPicker();
  if (PICK_AT !== null) loadPieceLibrary();
}

// The picker drawn again where it stands (and taken away where it does not), so the rest
// of the form, and the cursor in a field, is left alone.
function redrawPicker() {
  const form = document.getElementById('form');
  if (!form) return;
  form.querySelectorAll('[data-pickat]').forEach(box => {
    box.innerHTML = String(PICK_AT) === box.dataset.pickat ? pickerHtml(PICK_AT) : '';
  });
  const opener = form.querySelector('#dPickNew');
  if (opener) {
    opener.setAttribute('aria-expanded', String(PICK_AT === 'new'));
    opener.textContent = PICK_AT === 'new' ? 'close the piece library'
      : 'choose from the piece library';
  }
  form.querySelectorAll('[data-pickfor]').forEach(node => {
    const open = Number(node.dataset.pickfor) === PICK_AT;
    node.setAttribute('aria-expanded', String(open));
    node.textContent = open ? 'close the drawings' : 'choose a drawing';
  });
  wirePicker(form);
}

// Only the chips, the count and the tiles, so the cursor stays in the search box.
function redrawPickList() {
  const kinds = document.getElementById('dPickKinds');
  const list = document.getElementById('dPickList');
  const count = document.getElementById('dPickCount');
  if (kinds) kinds.innerHTML = pickKindsHtml(PICK_AT);
  if (list) list.innerHTML = pickListHtml(PICK_AT);
  if (count) count.textContent = pickCountWords(PICK_AT);
  const form = document.getElementById('form');
  if (form) wirePickList(form);
  placePickAbout();
}

function wirePickList(form) {
  form.querySelectorAll('[data-pickkind]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      PICK_KIND = node.dataset.pickkind;
      PICK_SHOWN = '';
      redrawPickList();
    };
  });
  form.querySelectorAll('[data-pick]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      togglePickAbout(node.dataset.pick);
    };
  });
  form.querySelectorAll('[data-palettepiece]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      takePalettePiece(node.dataset.palettepiece);
    };
  });
  const grid = form.querySelector('#dPickGrid');
  if (grid) {
    // Escape closes the open line and leaves the focus on the tile that opened it.
    grid.onkeydown = event => {
      if (event.key !== 'Escape' || !PICK_SHOWN) return;
      const slug = PICK_SHOWN;
      togglePickAbout(slug);
      const button = grid.querySelector(`[data-pick="${CSS.escape(slug)}"]`);
      if (button) button.focus();
    };
  }
  // The grid's columns follow its width, so the line moves to its tile's row on a resize.
  if (!wirePickList.resizing) {
    wirePickList.resizing = true;
    window.addEventListener('resize', () => { if (PICK_SHOWN) placePickAbout(); });
  }
}

function wirePicker(form) {
  const opener = form.querySelector('#dPickNew');
  if (opener) opener.onclick = event => { event.preventDefault(); openPicker('new'); };
  form.querySelectorAll('[data-pickfor]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      event.stopPropagation();
      openPicker(Number(node.dataset.pickfor));
    };
  });
  if (PICK_AT === null) return;
  const find = form.querySelector('#dPickFind');
  if (find) {
    find.oninput = () => { PICK_FIND = find.value; PICK_SHOWN = ''; redrawPickList(); };
  }
  const moving = form.querySelector('#dPickMoving');
  if (moving) {
    moving.onchange = () => { PICK_MOVING = moving.checked; PICK_SHOWN = ''; redrawPickList(); };
  }
  wirePickList(form);
  placePickAbout();
}

// A piece just added from the picker, brought into view with its card open, since on a
// phone the picker is below the list it joined.
function showNewPiece(index) {
  const card = document.querySelector(`.editorPiece[data-piece="${index}"]`);
  if (!card) return;
  if (card.scrollIntoView) card.scrollIntoView({ block: 'start' });
  const fold = card.querySelector('[data-fold]');
  if (fold && fold.focus) fold.focus({ preventScroll: true });
}

// A new piece, with the knight's leap every new piece arrives with, wearing the drawing
// named, if any, and called after it.
function addPieceWearing(slug) {
  const drawing = (PIECE_LIBRARY ? PIECE_LIBRARY.pieces : []).find(one => one.slug === slug);
  DRAFT.pieces = DRAFT.pieces || [];
  const piece = {
    id: freeLetter(''), name: drawing ? drawing.name : 'new piece', value: 1,
    moves: [{ kind: 'leap', vectors: [[1, 2]], symmetry: 'dihedral8' }],
  };
  DRAFT.pieces.push(piece);
  if (drawing) {
    DRAFT.appearance = DRAFT.appearance || {};
    DRAFT.appearance.icons = DRAFT.appearance.icons || {};
    DRAFT.appearance.icons[piece.id] = slug;
  }
  DRAFT_PIECE = DRAFT.pieces.length - 1;
  DRAFT_SQUARE = -1;
  PICK_SHOWN = '';
  freshPiece(DRAFT_PIECE);
  // On hexagons the knight's leap it arrives with is the hex knight's (Q640).
  if ((DRAFT.board || {}).lattice === 'hex') return ontoHex();
  drawDesigner();
  checkDraft();
  loadMoveMap();
  showNewPiece(DRAFT_PIECE);
}

// '@self' is the piece's letter and '@royals' the draft's royal pieces, wherever they
// stand in the rule the entry gives; everything else is copied as written.
function readyRules(rules, id) {
  const royals = (DRAFT.pieces || []).filter(piece => piece.royal).map(piece => piece.id);
  const fill = value => {
    if (value === '@self') return id;
    if (value === '@royals') return royals.slice();
    if (Array.isArray(value)) return value.map(fill);
    if (value && typeof value === 'object') {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, fill(item)]));
    }
    return value;
  };
  return fill(rules);
}

function takeReadyPiece(slug) {
  const drawing = (PIECE_LIBRARY ? PIECE_LIBRARY.pieces : []).find(one => one.slug === slug);
  const ready = drawing && drawing.ready;
  if (!ready) return;
  DRAFT.pieces = DRAFT.pieces || [];
  const piece = Object.assign(JSON.parse(JSON.stringify(ready.piece)),
                              { id: freeLetter(ready.id) });
  DRAFT.pieces.push(piece);
  DRAFT.appearance = DRAFT.appearance || {};
  DRAFT.appearance.icons = DRAFT.appearance.icons || {};
  DRAFT.appearance.icons[piece.id] = slug;
  if (ready.rules || ready.note) READY_NEEDS[piece.id] = ready;
  DRAFT_PIECE = DRAFT.pieces.length - 1;
  DRAFT_SQUARE = -1;
  PICK_SHOWN = '';
  freshPiece(DRAFT_PIECE);
  drawDesigner();
  checkDraft();
  loadMoveMap();
  showNewPiece(DRAFT_PIECE);
}

// On the card of a piece taken ready-made: what its tradition adds that the draft must
// say, a rule of the whole game it needs (with a button that writes it), or both.
function readyNeedsHtml(index) {
  const piece = (DRAFT.pieces || [])[index];
  const ready = piece && READY_NEEDS[piece.id];
  if (!ready) return '';
  const parts = [];
  if (ready.rules) {
    const rules = readyRules(ready.rules, piece.id);
    const held = DRAFT.rules || {};
    const keys = Object.keys(rules);
    const named = keys.map(key => 'rules.' + key).join(', ');
    const written = keys.every(key => JSON.stringify(held[key]) === JSON.stringify(rules[key]));
    const replaced = keys.filter(key => key in held).map(key => 'rules.' + key).join(', ');
    parts.push(written
      ? `<div>Its rule is written: <code>${esc(named)}</code>.</div>`
      : `<div><b>It needs a rule of the whole game.</b> ${esc(ready.rulesSay)}</div>
         <div class="row u-gap-8 u-mt-6"><button data-readyrules="${index}">write ${
           esc(named)} into the game</button>
         <code class="small">${esc(JSON.stringify(rules))}</code></div>${replaced
           ? `<div class="muted small">That replaces the game's own ${esc(replaced)}.</div>`
           : ''}`);
  }
  if (ready.note) parts.push(`<div>${esc(ready.note)}</div>`);
  return `<div class="freshNote readyNeeds small" data-readyneeds="${index}">${parts.join('')}
    <button class="u-mt-6" data-readydone="${index}">done with this note</button></div>`;
}

function wireReadyNeeds(form) {
  form.querySelectorAll('[data-readyrules]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const piece = DRAFT.pieces[Number(node.dataset.readyrules)];
      const ready = piece && READY_NEEDS[piece.id];
      if (!ready || !ready.rules) return;
      DRAFT.rules = Object.assign(DRAFT.rules || {}, readyRules(ready.rules, piece.id));
      drawDesigner();
      checkDraft();
    };
  });
  form.querySelectorAll('[data-readydone]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const piece = DRAFT.pieces[Number(node.dataset.readydone)];
      if (piece) delete READY_NEEDS[piece.id];
      drawDesigner();
    };
  });
}

// ------------------------------------------------ a piece's card, folded
//
// PF4 (the user, 2026-10-26). Each piece's card folds to one line -- its drawing, its name
// and its moves in the words a game's page says them in (the check's `move_lines`, out of
// `spec/describe.py`) -- and opens to the whole editor. Which cards are open is kept by the
// piece's letter for as long as the page is, so a redraw, an undo or a check leaves it;
// a piece just added opens.
const PIECES_OPEN = new Set();   // the letters of the pieces whose cards are open

function pieceFoldSays(index) {
  const piece = (DRAFT.pieces || [])[index] || {};
  const lines = MOVE_SAID && MOVE_SAID.key === JSON.stringify(DRAFT)
    ? (MOVE_SAID.lines || [])[index] : null;
  const said = (lines || []).filter(Boolean);
  const count = (piece.moves || []).length;
  const moves = said.length ? said.join('; ')
    : count ? `${count} way${count === 1 ? '' : 's'} of moving` : 'no way of moving';
  return (piece.royal ? 'royal; ' : '') + moves;
}

function setPieceOpen(index, open) {
  const card = document.querySelector(`.editorPiece[data-piece="${index}"]`);
  if (!card) return;
  card.classList.toggle('folded', !open);
  const fold = card.querySelector('[data-fold]');
  if (fold) fold.setAttribute('aria-expanded', String(open));
  const body = card.querySelector('.pieceBody');
  if (body) body.hidden = !open;
}

function wireFolds(form) {
  form.querySelectorAll('[data-fold]').forEach(node => {
    node.onclick = () => {
      const index = Number(node.dataset.fold);
      const piece = (DRAFT.pieces || [])[index];
      if (!piece) return;
      const open = !PIECES_OPEN.has(piece.id);
      if (open) PIECES_OPEN.add(piece.id); else PIECES_OPEN.delete(piece.id);
      setPieceOpen(index, open);
    };
  });
}

// The folded lines' words, once the check has said each piece's moves.
function drawFoldSays() {
  document.querySelectorAll('[data-foldsays]').forEach(node => {
    node.textContent = pieceFoldSays(Number(node.dataset.foldsays));
  });
}

// The pieces in the order the list shows them: the draft's (a portal's moves are its
// group's since spelling 31, on its card among the rules, so no piece here is a portal).
function piecesInOrder() {
  return (DRAFT.pieces || []).map((_, index) => index);
}

function pieceEditor(piece, index) {
  const on = index === DRAFT_PIECE;
  const open = PIECES_OPEN.has(piece.id);
  return `<div class="editorPiece${on ? ' on' : ''}${open ? '' : ' folded'}" data-piece="${index}">
    <button class="pieceFold" data-fold="${index}" aria-expanded="${open}"
      aria-controls="dPieceBody${index}"><span class="foldMark" aria-hidden="true"></span
      ><span class="iconSlot" data-icon="${index}">${pieceIconHtml(index)}</span
      ><span class="foldName" data-foldname="${index}">${esc(piece.name || piece.id || '')}</span
      ><span class="foldSays muted small" data-foldsays="${index}">${
        esc(pieceFoldSays(index))}</span></button>
    <div class="pieceBody" id="dPieceBody${index}"${open ? '' : ' hidden'}>
    <div class="row u-gap-8">
      <label class="field u-w-64 u-m-0"><span>letter</span>
        <input data-field="id" data-index="${index}" maxlength="2"
          value="${esc(piece.id || '')}"></label>
      <label class="field u-flex-1 u-minw-110 u-m-0"><span>name</span>
        <input data-field="name" data-index="${index}" value="${esc(piece.name || '')}"></label>
      <label class="field u-w-74 u-m-0"><span>worth</span>
        <input data-field="value" data-index="${index}" type="number" step="0.5"
          value="${esc(piece.value === undefined ? 1 : piece.value)}"></label>
      <label class="pick u-mt-14"><input type="checkbox" data-field="royal"
        data-index="${index}"${piece.royal ? ' checked' : ''}> royal${
        info('royal')}</label>
      <span class="muted small pieceSpeed" data-speedof="${index}">${
        esc(pieceSpeedWords(piece))}</span>
      <span class="muted small pieceSpeed" data-restof="${index}">${
        esc(pieceRestWords(piece))}</span>
    </div>
    <div class="row small u-mt-6 u-gap-8">
      <span data-uploads="${index}" class="u-d-contents">${uploadButtonsHtml(index)}</span>
      <input data-field="icon" data-index="${index}" placeholder="or name a library piece: knight, cannon..."
        title="one name serves both players; two names separated by a space draw one each"
        value="${esc(iconNameOf(piece.id))}" class="u-flex-1 u-minw-150 u-fs-12 u-p-5-8">
      <button data-pickfor="${index}" aria-expanded="${PICK_AT === index}"
        aria-controls="dPicker">${PICK_AT === index ? 'close the drawings' : 'choose a drawing'}</button>
    </div>
    <div data-pickat="${index}">${PICK_AT === index ? pickerHtml(index) : ''}</div>
    ${freshNoteHtml(index)}
    ${readyNeedsHtml(index)}
    <div class="u-mt-8">
      ${(piece.moves || []).map((move, m) => moveEditor(move, index, m)).join('')}
      ${sightsHtml(piece, index)}
      <div class="row u-mt-7">
        <button data-addmove="${index}">add a way of moving</button>
        ${sightAddHtml(piece, index)}
        <button data-drop="${index}" title="remove this piece from the game">remove piece</button>
      </div>
      ${pieceTurnsHtml(piece, index)}
      ${extrasHtml('piece.' + index, piece, 'more about this piece',
                   new Set(pieceRowKeys().rounds.concat(pieceRowKeys().always)))}
    </div>
    </div>
  </div>`;
}

// The board's size, and the lattice its cells sit on. Every control here writes through
// `writeBoardField`, which is what the round trip in `tests/test_designer.py` reads back.
//
// On the hex lattice the board is a rhombus or a hexagon (`board.shape`, schema 16). A
// hexagon is as tall as it is wide and odd, so it has one size, *cells on a side*, which
// writes the width and the height together; 2 to 8 on a side is the schema's 3 to 15.
// The shape menu is drawn wherever the draft states a shape, so a hexagon wrongly put on
// squares still shows what it says.
function boardFormHtml(board) {
  const lattice = board.lattice || 'square';
  const shape = board.shape || 'grid';
  const hexagon = shape === 'hexagon';
  const size = hexagon
    ? `<label class="field"><span>cells on a side</span>
        <input id="dSide" data-bfield="side" type="number" min="${HEXAGON_SIDES[0]}"
          max="${HEXAGON_SIDES[1]}" value="${esc(hexagonSide(board))}"></label>`
    : `<div class="pair">
      <label class="field"><span>board width</span>
        <input id="dWidth" data-bfield="width" type="number" min="2" max="16"
          value="${esc(board.width || 8)}"></label>
      <label class="field"><span>board height</span>
        <input id="dHeight" data-bfield="height" type="number" min="2" max="16"
          value="${esc(board.height || 8)}"></label>
    </div>`;
  const shapes = lattice === 'hex' || 'shape' in board
    ? `<label class="field"><span>the board is</span>
      <select id="dShape" data-bfield="shape">
        ${[['grid', 'a rhombus'], ['hexagon', 'a hexagon']].map(([value, label]) =>
          `<option value="${value}"${shape === value ? ' selected' : ''}>${label}</option>`
        ).join('')}
      </select></label>` : '';
  const note = lattice !== 'hex' ? ''
    : hexagon ? `<div class="muted small cardSays">A hexagon's corners are not cells at all,
        so no obstacles are needed to cut them and no piece can stand there. Its three
        pairs of opposite sides can each be joined on the wrap card. List every direction a
        piece moves in, because turning a direction is the square grid's.</div>`
    : `<div class="muted small cardSays">A hex board is a width by height
        rhombus of hexagons. Cut the shape you want out of it with obstacles, or make it a
        hexagon, and list every direction a piece moves in, because turning a direction is
        the square grid's.</div>`;
  const removed = BOARD_REMOVED
    ? `<div class="muted small cardSays u-mb-8" id="dBoardRemoved">${
      esc(BOARD_REMOVED)}</div>` : '';
  return `${size}${removed}
    <label class="field"><span>the cells are</span>
      <select id="dLattice" data-bfield="lattice">
        ${[['square', 'squares'], ['hex', 'hexagons']].map(([value, label]) =>
          `<option value="${value}"${lattice === value ? ' selected' : ''}>${label}</option>`
        ).join('')}
      </select>${note}</label>
    ${shapes}`;
}

// The board keys a control writes, where they are not the control's own name: a
// hexagon's one size is both of its frame's.
const BOARD_FIELD_KEYS = { side: ['width', 'height'] };

// The fewest and most cells a hexagon has on a side: `schema.py`'s odd width from 3 to 15.
const HEXAGON_SIDES = [2, 8];

// How many cells a side of this hexagon has, read off its width; a board that is not yet
// a hexagon gets the one nearest its larger side.
function hexagonSide(board) {
  const across = Math.max(Number(board.width) || 0, Number(board.height) || 0) || 11;
  return Math.min(HEXAGON_SIDES[1], Math.max(HEXAGON_SIDES[0], Math.floor((across + 1) / 2)));
}

// One board control written into the draft. A value the board already means is left as
// written, so loading a game and touching nothing changes nothing.
//
// `side` is the hexagon's one size and writes both of the frame's. Choosing the hexagon
// makes the frame one, and leaving the hex lattice leaves the hexagon too, since the
// validator allows it nowhere else. `grid` written out stays written, as `lattice` does.
function writeBoardField(board, field, value) {
  if (field === 'lattice') {
    if (value === (board.lattice || 'square')) return;
    if (value === 'square') delete board.lattice; else board.lattice = value;
    if (value !== 'hex' && board.shape === 'hexagon') delete board.shape;
    return;
  }
  if (field === 'shape') {
    if (value === (board.shape || 'grid')) return;
    if (value === 'grid') { delete board.shape; return; }
    board.shape = value;
    writeBoardField(board, 'side', hexagonSide(board));
    return;
  }
  if (field === 'side') {
    // An emptied box keeps the size it had; any number is held to the schema's range.
    const asked = String(value).trim() === '' ? NaN : Math.round(Number(value));
    const side = Math.min(HEXAGON_SIDES[1],
      Math.max(HEXAGON_SIDES[0], Number.isFinite(asked) ? asked : hexagonSide(board)));
    const across = 2 * side - 1;
    if (board.width !== across) board.width = across;
    if (board.height !== across) board.height = across;
    return;
  }
  const size = Number(value) || 8;
  if (board[field] !== size) board[field] = size;
}

// The named squares a board change left off the board, taken out of the draft, and what
// was taken, in words (Q281c3). Which squares those are is the server's (`offBoard` on a
// check, `designing.off_board`): each entry is a list's path and the names to take out of
// it, so nothing here knows which squares a board has. A wall list or an army row left
// empty goes with its last square, since the validator refuses an empty one.
function dropOffBoard(draft, offBoard) {
  const said = [];
  const emptied = [];
  let why = '';
  (offBoard || []).forEach(({ path, squares, why: reason }) => {
    const owner = path.slice(0, -1).reduce((at, key) => (at == null ? at : at[key]), draft);
    const key = path[path.length - 1];
    if (!owner || !Array.isArray(owner[key])) return;
    const gone = new Set(squares);
    const taken = owner[key].filter(name => gone.has(name));
    if (!taken.length) return;
    owner[key] = owner[key].filter(name => !gone.has(name));
    why = why || reason;
    const where = taken.join(', ');
    if (path[0] === 'setup') said.push(`the ${owner.piece || 'piece'} at ${where}`);
    else if (path[1] === 'obstacles') said.push(`the walls at ${where}`);
    else said.push(`the promotion squares ${where}`);
    if (owner[key].length) return;
    if (path[0] === 'setup') emptied.push(owner);
    if (path[1] === 'obstacles') delete draft.rules.obstacles;
  });
  if (emptied.length) {
    draft.setup.placements = draft.setup.placements.filter(row => !emptied.includes(row));
  }
  return said.length ? `The board change took away ${said.join('; ')}: ${why}.` : '';
}

// A check's answer after a board control changed, and only then: the named squares that
// change left off the board go, the form says which, and the draft is checked again
// without them. True when it did, so the caller drops this now-stale answer. The one
// path from a check to a redraw of the form, and never taken on a keystroke elsewhere.
function fitAfterReshape(answer) {
  if (!BOARD_RESHAPED) return false;
  BOARD_RESHAPED = false;
  const removed = answer.valid ? '' : dropOffBoard(DRAFT, answer.offBoard);
  if (!removed) return false;
  BOARD_REMOVED = removed;
  drawDesigner();
  checkDraft();
  return true;
}

// The wrap keys the draft's shape does not take, dropped when the shape changes. A grid's
// `files` means nothing on a hexagon, whose sides are joined pair by pair, and a pair means
// nothing on a grid; the validator refuses each on the other shape. Which keys are whose is
// the server's (`cards.shapeKeys`), never a list kept here.
function fitWrapToShape() {
  const wrap = (DRAFT.rules || {}).wrap;
  const keys = ((cardModel() || {}).shapeKeys || {})['rules.wrap'];
  if (!wrap || typeof wrap !== 'object' || !keys) return;
  const shape = draftShape();
  Object.entries(keys).filter(([other]) => other !== shape)
    .forEach(([, owned]) => owned.forEach(key => { delete wrap[key]; }));
  if (!Object.keys(wrap).length) delete DRAFT.rules.wrap;
}

// The draft just made hexagons, with what only squares can read rewritten by the server
// (`designing.onto_hex`): each move keeps its vectors and takes the hex lattice's turns
// and mirror for the square's, the armies face each other by the half turn, a rank zone is
// named cell by cell. What it changed is said under the board controls, and undo takes it
// back with the lattice. A draft edited again before the answer is left alone; a failed
// request leaves the draft as written, for the check to name what squares still hold.
function ontoHex() {
  const asked = JSON.stringify(DRAFT);
  return api('/api/design/hex', { spec: DRAFT }).then(answer => {
    if (JSON.stringify(DRAFT) !== asked) return;
    if (answer && answer.spec) DRAFT = answer.spec;
    BOARD_REMOVED = (answer && answer.said) || '';
  }).catch(() => {}).then(() => {
    drawDesigner();
    checkDraft();
    loadMoveMap();
  });
}

// One way of moving: how it travels, in which directions, which of those directions are
// mirrored for it, how far, how near it may stop, which rank it must start from, and
// whether it may move there, take there, or both.
//
// **Every control carries its label** (option A6). The row used to be four menus and
// three unlabelled number boxes reading *far*, *1* and *any*, which are placeholders and
// look like values already entered, with the meaning of each in a tooltip. Nothing on the
// row said which of a direction's two numbers is sideways, either.
//
// The kinds are the schema's `MOVE_KINDS`, which a test holds this menu to. A sweep takes
// a slide's fields, so it needs no control of its own.
const TURNINGS = [['none', 'as written'], ['lateral', 'left and right'],
                  ['rotate4', 'four ways'], ['dihedral8', 'all eight ways'],
                  ['rotate6', 'six ways'], ['dihedral12', 'all twelve ways']];
const SQUARE_TURNS = ['none', 'lateral', 'rotate4', 'dihedral8'];
const HEX_TURNS = ['none', 'rotate6', 'dihedral12'];

function onHexagons() {
  return ((DRAFT && DRAFT.board) || {}).lattice === 'hex';
}

// The *turned* menu's choices. Each lattice has its own turns (schema 18b): the square's
// quarter turns carry a direction off a hexagon's cells and the hexagon's sixth turns
// carry one off a square's, which the rules refuse either way. So a draft is offered its
// own lattice's, plus a move's own turn while it still has one, so the menu shows what is
// there and setting it back is one choice.
function turnings(move) {
  const own = onHexagons() ? HEX_TURNS : SQUARE_TURNS;
  return TURNINGS.filter(([value]) => own.includes(value) || value === move.symmetry);
}

// Q1007a (`docs/plan-schema-31.md` §2.5): a jump longer than one square may be ridden, jump
// after jump in its direction. Offered on a leap, a slide or a blockable leap whose every
// direction is longer than one square, as one menu read off the descriptor: once (a leap, or
// a blockable leap of one jump), again and stopped only where it lands (a slide over the jump,
// the nightrider), or again and also stopped on the square each jump passes (a blockable leap
// with a range, the lame rider). Writing the reading it already has changes nothing.
const RIDES = [['once', 'no, one jump'], ['lands', 'yes, stopped only where it lands'],
               ['passes', 'yes, also stopped on the square each jump passes']];

function ridable(move) {
  const vectors = move.vectors || [];
  return ['leap', 'slide', 'lame'].includes(move.kind || 'slide') && vectors.length > 0
    && vectors.every(v => Math.max(Math.abs(v[0]), Math.abs(v[1])) > 1);
}

function rideOf(move) {
  const kind = move.kind || 'slide';
  if (kind === 'slide') return move.range === 1 ? 'once' : 'lands';
  if (kind === 'lame') return (move.range === undefined ? 1 : move.range) === 1 ? 'once' : 'passes';
  return 'once';
}

function writeRide(move, value) {
  if (value === rideOf(move)) return;
  const kind = move.kind || 'slide';
  if (value === 'once') {
    if (kind === 'slide') move.kind = 'leap';
    delete move.range;
    delete move.min_range;
  } else if (value === 'lands') {
    move.kind = 'slide';
    if (kind !== 'lame' || move.range === 0) delete move.range;
    if (kind === 'leap') delete move.min_range;
  } else if (value === 'passes') {
    move.kind = 'lame';
    if (kind !== 'slide' || move.range === undefined || move.range === 1) move.range = 0;
    if (kind === 'leap') delete move.min_range;
  }
}

function moveEditor(move, piece, index) {
  const field = (label, control, hint) => `<label class="moveField"
    title="${esc(hint || '')}"><span>${esc(label)}</span>${control}</label>`;
  const pick = (key, options, chosen) => `
    <select data-move="${piece}.${index}" data-mfield="${key}">
      ${options.map(([value, label]) => `<option value="${esc(value)}"${
        value === chosen ? ' selected' : ''}>${esc(label)}</option>`).join('')}
    </select>`;
  const box = (key, value, placeholder, attrs) => `<input data-move="${piece}.${index}"
    data-mfield="${key}" value="${value}" placeholder="${esc(placeholder)}" ${attrs || ''}>`;
  // The row starts short (Q545, the user's *"A"*, 2026-10-01): how it travels, which way,
  // turned how, and whether it moves or takes are what most moves need. How far, how near
  // and from which rank fold under *more*, open whenever one of them says something.
  const condition = move.condition || {};
  const more = move.range !== undefined || move.min_range !== undefined
    || condition.from_rank !== undefined;
  // A portal group's rows (owner `p<n>`, spelling 31) offer only what a portal may do (D1):
  // no sweep, *move only* (what a group's move means with no mode written), the turns that
  // read the same for both players, and no rank to move from. A value already written stays
  // on its menu, so the row shows what the draft says beside the refusal of it.
  const portal = isPortalKey(piece);
  const kinds = [['slide', 'slides'], ['leap', 'leaps'], ['hop', 'hops over'],
                 ['lame', 'blockable leap'], ['sweep', 'sweeps']]
    .filter(([value]) => !portal || value !== 'sweep' || move.kind === 'sweep');
  const mode = move.mode || (portal ? 'move' : 'both');
  const modes = [['both', 'move or take'], ['move', 'move only'], ['capture', 'take only']]
    .filter(([value]) => !portal || value === 'move' || value === mode);
  const turns = turnings(move).filter(([value]) => !portal || ['rotate4', 'dihedral8']
    .includes(value) || value === (move.symmetry || 'none'));
  return `<div class="moveRow">
    ${field('how it travels', pick('kind', kinds, move.kind || 'slide'),
      'slides run until something stops them; leaps jump; hops need a piece to jump'
      + ' over; sweeps jump along the line and take everything they pass, their own'
      + ' pieces too')}
    ${ridable(move) && !portal ? field('jumps again', pick('ride', RIDES, rideOf(move)),
      'whether it may jump again and again in the same direction, as a nightrider does; a'
      + ' rider is stopped by a piece where it would land, and may also be stopped by one on'
      + ' the square each jump passes, as a xiangqi horse is') : ''}
    ${field('turned', pick('symmetry', turns, move.symmetry || 'none'),
      onHexagons() ? 'which of the hexagon\'s turns the direction also gets: six ways, or'
        + ' six ways and their mirrors'
        : 'which turns of the direction it also gets')}
    ${field('direction, sideways then forward',
      box('vectors', esc((move.vectors || []).map(v => v.join(',')).join(' ')), '0,1 1,2'),
      'one step as sideways,forward -- 0,1 is one square forward and 1,0 is one to the'
      + ' right; several separated by spaces')}
    ${field('and it may', pick('mode', modes, mode),
      'a pawn moves one way and takes another, which is two rows')}
    <button data-dropmove="${piece}.${index}" title="remove this move">&times;</button>
    <details class="moveMore"${more ? ' open' : ''}>
      <summary class="muted small" title="how far, how near, and from which rank">more</summary>
      <div class="moveMoreBody">
    ${field('how far',
      box('range', esc(move.range), 'to the edge', 'type="number" min="0" max="16"'),
      'how many squares (jumps, for one that jumps again); blank runs to the edge, or is one'
      + ' jump for a blockable leap')}
    ${field('no nearer than',
      box('min_range', esc(move.min_range), '1', 'type="number" min="1" max="16"'),
      "the nearest square it may stop on; 2 is a pawn's double step, which must go two")}
    ${portal && condition.from_rank === undefined ? '' : field('only from rank',
      box('from_rank', condition.from_rank === undefined
            ? '' : esc(Number(condition.from_rank) + 1),
          'any', 'type="number" min="1" max="16"'),
      "only from this rank, counted from the player's own side; blank is from anywhere")}
      </div>
    </details>
  </div>
  <div class="moveSays muted small" data-movesays="${piece}.${index}">${
    esc(moveSaid(piece, index))}</div>
  ${extrasHtml('move.' + piece + '.' + index, move, 'more about this move',
               moveExtrasSkip(move))}`;
}

// What each typed row says, in the words the game's own page uses (F10: a step typed as
// *1,1* had no read-back of what it does). The sentences are the check's (`move_lines`),
// held with the draft they were said of, so a row is never read back from another draft:
// between an edit and its check the old sentence stands, and a redraw that moved the rows
// shows none until the check answers.
let MOVE_SAID = null;   // { key, lines } -- the check's sentences and the draft they are of

function moveSaid(piece, index) {
  if (!MOVE_SAID || MOVE_SAID.key !== JSON.stringify(DRAFT)) return '';
  const lines = isPortalKey(piece) ? (MOVE_SAID.portalLines || [])[Number(String(piece).slice(1))]
    : (MOVE_SAID.lines || [])[piece];
  const said = (lines || [])[index] || '';
  return said ? 'That reads: ' + said + '.' : '';
}

function drawMoveSays() {
  if (!MOVE_SAID || MOVE_SAID.key !== JSON.stringify(DRAFT)) return;
  document.querySelectorAll('[data-movesays]').forEach(node => {
    const [piece, index] = node.dataset.movesays.split('.');
    node.textContent = moveSaid(isPortalKey(piece) ? piece : Number(piece), Number(index));
  });
}

// One move-row control written into its descriptor. A value the descriptor already means
// is left as written: `"mode": "both"` stays spelled out, and an absent `symmetry` is not
// written as `"none"`.
function writeMoveField(move, field, value) {
  if (field === 'vectors') {
    const vectors = value.trim().split(/\s+/).filter(Boolean)
      .map(pair => pair.split(',').map(Number))
      .filter(pair => pair.length === 2 && pair.every(n => Number.isFinite(n)));
    if (JSON.stringify(vectors) !== JSON.stringify(move.vectors)) move.vectors = vectors;
  } else if (field === 'range') {
    const range = value === '' ? undefined : Number(value);
    if (range === move.range) return;
    if (range === undefined) delete move.range; else move.range = range;
  } else if (field === 'min_range') {
    const near = Number(value) > 1 ? Number(value) : 1;
    if (near === (move.min_range || 1)) return;
    if (near > 1) move.min_range = near; else delete move.min_range;
  } else if (field === 'from_rank') {
    // Shown counted from one, as the board labels ranks; the spec counts from zero.
    const rank = Number(value) >= 1 ? Number(value) - 1 : undefined;
    if (rank === (move.condition || {}).from_rank) return;
    const condition = Object.assign({}, move.condition);
    if (rank === undefined) delete condition.from_rank; else condition.from_rank = rank;
    if (Object.keys(condition).length) move.condition = condition; else delete move.condition;
  } else if (field === 'ride') {
    writeRide(move, value);
  } else if (field === 'symmetry') {
    if (value === (move.symmetry || 'none')) return;
    if (value === 'none') delete move.symmetry; else move.symmetry = value;
  } else if (field === 'mode') {
    if (value !== (move.mode || 'both')) move.mode = value;
  } else if (value !== move[field]) {
    move[field] = value;
  }
}

// ------------------------------------------------- lines of sight (schema 30, Q999c)
//
// A piece's lines of sight (`docs/archive/plan-passive.md` §4.7): one row each under its ways of
// moving, built from the move row's own pickers (direction, turned, how far, no nearer
// than) and four of its own: what it does, to whose pieces, whether it sees through pieces
// (an aura) or stops at the first (a ray), and which kinds of piece it touches. Every label
// and its sentence are `spec.plain`'s (`cards.sight_kit`), so a refusal that names a field
// names it as the row does; the refusals are the validator's, said under the rows
// (`sightProblems`), and each row is read back in the words a game's page says it in
// (`sightLines`). The menus offer the schema's values (`kit.effects`, `kit.whose`), and a
// value already written stays on its menu, so the row shows what the draft says beside the
// refusal of it.
const SIGHT_EFFECT_WORDS = { freeze: 'freezes it', disarm: 'disarms it', shield: 'shields it',
                             command: 'commands it' };
const SIGHT_WHOSE_WORDS = { enemy: "the other player's", own: "its own side's",
                            any: "both sides'" };
const SIGHT_LINE_WORDS = [['ray', 'no: stops at the first piece'], ['aura', 'yes']];
// The line of sight a piece is given: Ultima's immobiliser, freezing every enemy piece on
// the eight squares around it, which is the plan's own first example.
const NEW_SIGHT = { vectors: [[1, 0], [1, 1]], symmetry: 'rotate4', range: 1,
                    effect: 'freeze', whose: 'enemy' };

// Generation 37 (N2, `docs/plan-npc.md` §4.2): a wild piece's type may see too, with its
// three limits: both sides' pieces (`whose` 'any', a wild piece has no side), no command
// (nobody commands for it), and lines that read the same for both players (the validator
// says so in words; the ring below does). Its new line of sight is the ring freezing both
// sides' pieces.
const NEW_WILD_SIGHT = { vectors: [[1, 0], [1, 1]], symmetry: 'rotate4', range: 1,
                         through: true, effect: 'freeze', whose: 'any' };

function isWildPiece(index) {
  const piece = (DRAFT.pieces || [])[index];
  return !!piece && ((DRAFT.rules || {}).npcs || []).some(npc => npc && npc.piece === piece.id);
}

function sightKit() {
  return (cardModel() || {}).sight || { max: 4, effects: [], whose: [], words: {} };
}

function sightWords(key) {
  return sightKit().words[key] || [key.replace(/_/g, ' '), ''];
}

// A menu's options, with the value the draft holds kept on it when it is none of them.
function withHeld(options, held) {
  return held === undefined || options.some(([value]) => value === held)
    ? options : options.concat([[held, String(held)]]);
}

// The pieces a sight may name in *only these pieces*: every piece of the draft but the wild
// pieces, which nothing affects, and any it already names.
function sightTargets(sight) {
  const wild = new Set(((DRAFT.rules || {}).npcs || []).map(npc => npc && npc.piece));
  const named = Array.isArray(sight.affects) ? sight.affects : [];
  // Q1007d: a royal is never commanded (the validator refuses one named), so a command's row
  // offers none, unless the draft already names one, which stays beside its refusal.
  const command = sight.effect === 'command';
  return (DRAFT.pieces || [])
    .filter((piece, index) => piece && piece.id && (named.includes(piece.id)
      || (!wild.has(piece.id) && !(command && piece.royal === true))))
    .map(piece => [piece.id, piece.name || piece.id]);
}

function sightEditor(sight, piece, index) {
  const tag = `data-sight="${piece}.${index}"`;
  const field = (key, control) => `<label class="moveField" data-sfor="${key}"
    title="${esc(sightWords(key)[1])}"><span>${esc(sightWords(key)[0])}</span>${control}</label>`;
  const pick = (key, options, chosen) => `
    <select ${tag} data-sfield="${key}">
      ${options.map(([value, label]) => `<option value="${esc(value)}"${
        value === chosen ? ' selected' : ''}>${esc(label)}</option>`).join('')}
    </select>`;
  const box = (key, value, placeholder, attrs) => `<input ${tag} data-sfield="${key}"
    value="${value}" placeholder="${esc(placeholder)}" ${attrs || ''}>`;
  const kit = sightKit();
  const wild = isWildPiece(piece);
  const effects = withHeld(kit.effects.filter(value => !wild || value !== 'command')
    .map(value => [value, SIGHT_EFFECT_WORDS[value] || value]), sight.effect);
  const whose = withHeld(kit.whose.filter(value => !wild || value === 'any')
    .map(value => [value, SIGHT_WHOSE_WORDS[value] || value]), sight.whose);
  const lines = sight.through === undefined ? 'ray'
    : sight.through === true ? 'aura' : String(sight.through);
  const named = Array.isArray(sight.affects) ? sight.affects : [];
  const more = sight.range !== undefined || sight.min_range !== undefined
    || sight.affects !== undefined;
  const affects = sightTargets(sight).map(([id, name]) => `<label class="pick sightPick">
      <input type="checkbox" ${tag} data-sfield="affects" data-pid="${esc(id)}"${
      named.includes(id) ? ' checked' : ''}> ${esc(name)}</label>`).join('');
  // Q1007d (`docs/plan-schema-31.md` §3.9): a command may make its pieces take their own side
  // (`captures_own`), a box beside the effect shown only while the effect is a command; ticked
  // writes `true`, unticked no key (`false` is the off value the schema takes as no key).
  const turncoat = sight.effect !== 'command' ? '' : `<label class="pick sightPick sightOwn"
      data-sfor="captures_own" title="${esc(sightWords('captures_own')[1])}"><input
      type="checkbox" ${tag} data-sfield="captures_own"${sight.captures_own === true
      ? ' checked' : ''}> ${esc(sightWords('captures_own')[0])}</label>`;
  return `<div class="moveRow sightRow">
    ${field('effect', pick('effect', effects, sight.effect))}
    ${turncoat}
    ${field('whose', pick('whose', whose, sight.whose))}
    ${field('vectors', box('vectors',
      esc((Array.isArray(sight.vectors) ? sight.vectors : [])
        .map(v => (Array.isArray(v) ? v : [v]).join(',')).join(' ')), '1,0 1,1'))}
    ${field('symmetry', pick('symmetry', turnings(sight), sight.symmetry || 'none'))}
    ${field('through', pick('through', withHeld(SIGHT_LINE_WORDS, lines), lines))}
    <button data-dropsight="${piece}.${index}" title="remove this line of sight">&times;</button>
    <details class="moveMore"${more ? ' open' : ''}>
      <summary class="muted small" title="how far, how near, and which pieces">more</summary>
      <div class="moveMoreBody">
    ${field('range', box('range', esc(sight.range), 'to the edge',
      'type="number" min="0" max="16"'))}
    ${field('min_range', box('min_range', esc(sight.min_range), '1',
      'type="number" min="1" max="16"'))}
    <div class="sightAffects" title="${esc(sightWords('affects')[1])}"><span
      class="muted small">${esc(sightWords('affects')[0])} (none ticked: every kind)</span>
      <span class="sightPicks">${affects}</span></div>
      </div>
    </details>
  </div>
  <div class="moveSays muted small" data-sightsays="${piece}.${index}">${
    esc(sightSaid(piece, index))}</div>`;
}

// A piece's lines of sight under its ways of moving, headed in `spec.plain`'s words, with
// the refusals that are about them; nothing for a piece that sees nothing.
function sightsHtml(piece, index) {
  if (!('sight' in piece)) return '';
  const rows = Array.isArray(piece.sight) ? piece.sight : [];
  return `<div class="sightRows" data-sights="${index}">
    <div class="sightHead small"><b>${esc(sightWords('sight')[0])}</b> <span class="muted">${
      esc(sightWords('sight')[1])}</span></div>
    ${rows.map((sight, k) => sightEditor(sight && typeof sight === 'object' ? sight : {},
                                         index, k)).join('')}
    <div class="sightProblems" data-sightproblems="${index}">${sightProblemsHtml(index)}</div>
  </div>`;
}

// *Add a line of sight*, beside *add a way of moving*, while the piece has room for one.
function sightAddHtml(piece, index) {
  const rows = Array.isArray(piece.sight) ? piece.sight.length : 0;
  const full = rows >= sightKit().max;
  return `<button data-addsight="${index}"${full ? ' disabled' : ''} title="${full
    ? `a piece sees along ${sightKit().max} lines at most`
    : 'what it sees: pieces on its lines are frozen, disarmed, shielded or commanded while they stand there'}"
    >add a line of sight</button>`;
}

// The check's sentences for each row, and its refusals for each piece, held with the draft
// they are of, as the move rows' are (`MOVE_SAID`).
let SIGHT_SAID = null;   // { key, lines, problems }

function sightSaid(piece, index) {
  if (!SIGHT_SAID || SIGHT_SAID.key !== JSON.stringify(DRAFT)) return '';
  const said = ((SIGHT_SAID.lines || [])[piece] || [])[index] || '';
  return said ? 'That reads: ' + said + '.' : '';
}

function sightProblemsHtml(piece) {
  if (!SIGHT_SAID || SIGHT_SAID.key !== JSON.stringify(DRAFT)) return '';
  return ((SIGHT_SAID.problems || [])[piece] || [])
    .map(line => `<div class="error small sightError">${esc(line)}</div>`).join('');
}

function drawSightSays() {
  if (!SIGHT_SAID || SIGHT_SAID.key !== JSON.stringify(DRAFT)) return;
  document.querySelectorAll('[data-sightsays]').forEach(node => {
    const [piece, index] = node.dataset.sightsays.split('.').map(Number);
    node.textContent = sightSaid(piece, index);
  });
  document.querySelectorAll('[data-sightproblems]').forEach(node => {
    node.innerHTML = sightProblemsHtml(Number(node.dataset.sightproblems));
  });
}

// One sight-row control written into its entry, as `writeMoveField` writes a move's: a value
// the entry already means is left as written, so a line of sight read in from a file and
// back out through the row is the file. `affects` takes the ids ticked, in the row's order.
function writeSightField(sight, field, value) {
  if (field === 'vectors') {
    const vectors = String(value).trim().split(/\s+/).filter(Boolean)
      .map(pair => pair.split(',').map(Number))
      .filter(pair => pair.length === 2 && pair.every(n => Number.isFinite(n)));
    if (JSON.stringify(vectors) !== JSON.stringify(sight.vectors)) sight.vectors = vectors;
  } else if (field === 'symmetry') {
    if (value === (sight.symmetry || 'none')) return;
    if (value === 'none') delete sight.symmetry; else sight.symmetry = value;
  } else if (field === 'range') {
    const range = value === '' ? undefined : Number(value);
    if (range === sight.range || (range === undefined && sight.range === undefined)) return;
    if (range === undefined) delete sight.range; else sight.range = range;
  } else if (field === 'min_range') {
    const near = Number(value) > 1 ? Number(value) : 1;
    if (near === (sight.min_range || 1)) return;
    if (near > 1) sight.min_range = near; else delete sight.min_range;
  } else if (field === 'through') {
    // A ray is the absent key; an aura is `true`, the one spelling the schema takes.
    if (value === 'aura') sight.through = true;
    else if (value === 'ray') delete sight.through;
  } else if (field === 'captures_own') {
    // Q1007d: ticked is `true`, unticked no key; a file's `false` (the off value) unticked is
    // left as written, so it round-trips.
    if (value === true) sight.captures_own = true;
    else if (value === false && sight.captures_own === true) delete sight.captures_own;
  } else if (field === 'effect') {
    // A `captures_own` means something only beside a command, so it goes with it.
    if (value !== sight.effect) sight.effect = value;
    if (value !== 'command') delete sight.captures_own;
  } else if (field === 'affects') {
    const ticked = Array.isArray(value) ? value : [];
    const held = Array.isArray(sight.affects) ? sight.affects : [];
    const same = ticked.length === held.length && ticked.every(id => held.includes(id));
    if (same && sight.affects !== undefined) return;
    if (ticked.length) sight.affects = ticked.slice(); else delete sight.affects;
  } else if (value !== sight[field]) {
    sight[field] = value;
  }
}

// The sight rows' controls, wired as the move rows' are; a change is checked, and a row
// added or taken away redraws the form.
function wireSights(form, changed) {
  form.querySelectorAll('[data-sight]').forEach(node => {
    const [piece, index] = node.dataset.sight.split('.').map(Number);
    const field = node.dataset.sfield;
    const write = () => {
      const sight = DRAFT.pieces[piece].sight[index];
      const value = field === 'captures_own' ? node.checked : field !== 'affects' ? node.value
        : [...form.querySelectorAll(`[data-sight="${piece}.${index}"][data-sfield="affects"]`)]
          .filter(box => box.checked).map(box => box.dataset.pid);
      const commanded = sight.effect === 'command';
      writeSightField(sight, field, value);
      DRAFT_PIECE = piece;
      // A change to or from a command shows or hides its `captures_own` box: the form redraws.
      if (field === 'effect' && commanded !== (sight.effect === 'command')) return changed(true);
      checkDraft();
    };
    node.onchange = write;
    if (node.tagName === 'INPUT' && node.type !== 'checkbox') node.oninput = write;
  });
  form.querySelectorAll('[data-addsight]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const piece = DRAFT.pieces[Number(node.dataset.addsight)];
      piece.sight = Array.isArray(piece.sight) ? piece.sight : [];
      const wild = isWildPiece(Number(node.dataset.addsight));
      piece.sight.push(JSON.parse(JSON.stringify(wild ? NEW_WILD_SIGHT : NEW_SIGHT)));
      DRAFT_PIECE = Number(node.dataset.addsight);
      // On hexagons it looks the hexagons' six ways, as a new way of moving goes (Q640).
      if ((DRAFT.board || {}).lattice === 'hex') return ontoHex();
      changed(true);
    };
  });
  form.querySelectorAll('[data-dropsight]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const [piece, index] = node.dataset.dropsight.split('.').map(Number);
      const held = DRAFT.pieces[piece];
      held.sight.splice(index, 1);
      // The last one taken away takes the key with it: `"sight": []` sees nothing and is
      // refused, so an empty list is never left behind.
      if (!held.sight.length) delete held.sight;
      changed(true);
    };
  });
}

function placementEditor(placement, index) {
  const ids = (DRAFT.pieces || []).map(p => p.id);
  // Whose row this is, which only means something without a mirror: under one, the
  // second player's army is the first player's reflected and the schema refuses a player.
  const sides = (lastPreview() || {}).sides || ['first', 'second'];
  const whose = (DRAFT.setup || {}).mirror === false
    ? `<select data-place="${index}" data-pfield="player" class="u-w-88">
        ${[0, 1].map(player => `<option value="${player}"${(placement.player || 0) === player
          ? ' selected' : ''}>${esc(sides[player])}</option>`).join('')}
      </select>`
    : '';
  return `<div class="row u-gap-7 u-mb-6">
    ${whose}
    <select data-place="${index}" data-pfield="piece" class="u-w-88">
      ${ids.map(id => `<option value="${esc(id)}"${id === placement.piece ? ' selected' : ''}
        >${esc(id)}</option>`).join('')}
    </select>
    <input data-place="${index}" data-pfield="squares" class="u-flex-1 u-minw-170"
      value="${esc((placement.squares || []).join(' '))}" placeholder="a1 b1 c1">
    <button data-dropplace="${index}" title="remove this row">&times;</button>
  </div>
  ${extrasHtml('place.' + index, placement, 'more about this row')}`;
}

// The slot beside a piece's fields. One drawing where one serves both players, and both
// where they differ -- otherwise an upload onto the second seat lands in the spec and
// shows up nowhere.
function pieceIconHtml(index) {
  const art = (lastPreview() || {}).icons || [];
  const pair = art[index] || [];
  const shown = (iconSeats(index) ? pair : pair.slice(0, 1)).filter(Boolean);
  if (!shown.length) {
    const piece = (DRAFT.pieces || [])[index] || {};
    return `<span class="muted">${esc(piece.id || '?')}</span>`;
  }
  return shown.map(icon => `<img${shown.length > 1 ? ' class="half u-tf-rot180"' : ''}
    src="${esc(icon.href)}" alt=""${icon.rotate
    ? '' : ''}>`).join('');
}

// Whether this piece is drawn differently per player, and what to call the two seats if
// it is. Two things ask for a pair: a facing, because a piece that points cannot also
// carry its side by being drawn upside down, and an entry that is already a pair
// because somebody typed two names into the field. `null` for everything else, where
// one drawing serves both.
//
// The names come from the check's own `sides` -- white and black for a two-colour set,
// first and second for one army turned round -- which is what the board calls them.
function iconSeats(index) {
  const preview = lastPreview() || {};
  const piece = (DRAFT.pieces || [])[index] || {};
  const named = ((DRAFT.appearance || {}).icons || {})[piece.id];
  const paired = Array.isArray(named) && named.length === 2;
  if (!paired && !(preview.directional || [])[index]) return null;
  return preview.sides || ['first', 'second'];
}

// The upload buttons under `root`, which is the whole form or one piece's button box:
// the box is rebuilt whenever a check changes whether the piece is drawn per player, so
// the buttons in it are new nodes and want wiring again.
function wireUploads(root) {
  root.querySelectorAll('[data-upload]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const seat = node.dataset.seat;
      uploadIcon(Number(node.dataset.upload), seat === undefined ? null : Number(seat));
    };
  });
}

// One upload button, or one per seat. An upload used to write a single string over the
// whole entry, so a drawing meant for one side of a facing piece became the drawing for
// both -- and the pair the game needs was gone.
//
// A server that withholds the upload route gets a sentence in the buttons' place.
function uploadButtonsHtml(index) {
  if (!offered('POST', '/api/design/icon')) {
    return `<span class="muted">This site does not take your own drawings yet, because an
      uploaded file has to be checked first. Name a library piece instead.</span>`;
  }
  const seats = iconSeats(index);
  if (!seats) return `<button data-upload="${index}">use my own drawing</button>`;
  return seats.map((side, seat) => `<button data-upload="${index}" data-seat="${seat}"
    title="this piece is drawn its own way for each player">my drawing for ${
    esc(side)}</button>`).join('');
}

// `appearance.icons[id]` is a library name, or a *pair* of them, one per player -- which
// a piece with a facing needs, because a piece that points cannot also carry its side by
// being drawn upside down. The field takes the pair back the way it shows it, as two
// names separated by a space. An uploaded drawing sits in the same slot as a data URI
// and has no name to show; sixty kilobytes of one in a text field would be worse than
// showing nothing, so it reads as blank.
function iconNameOf(id) {
  const named = ((DRAFT.appearance || {}).icons || {})[id];
  if (!named) return '';
  const names = (Array.isArray(named) ? named : [named])
    .map(entry => (String(entry).startsWith('data:') ? '' : String(entry)));
  return (names[0] === names[1] ? names[0] : names.join(' ')).trim();
}

function drawingHref(drawing) {
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(drawing.svg);
}

// A drawing picked: the piece wears it. A piece drawn its own way for each player (a
// facing, or a pair already typed) is given the drawing for both, which a two-colour
// drawing serves as its white and its black.
function wearDrawing(index, slug) {
  const piece = (DRAFT.pieces || [])[index];
  if (!piece) return;
  DRAFT.appearance = DRAFT.appearance || {};
  DRAFT.appearance.icons = DRAFT.appearance.icons || {};
  DRAFT.appearance.icons[piece.id] = iconSeats(index) ? [slug, slug] : slug;
  const field = document.querySelector(`[data-field="icon"][data-index="${index}"]`);
  if (field) field.value = iconNameOf(piece.id);
  // The tiles say which drawing it wears now, and the open line's button that it does.
  if (document.getElementById('dPickList')) {
    redrawPickList();
    const tile = document.querySelector(`[data-pick="${CSS.escape(slug)}"]`);
    if (tile) tile.focus();
  }
  checkDraft();
}

// Every control that lists the draft's pieces, drawn again from the draft. The form is
// redrawn whole on a structural change and never on a keystroke, so a typed letter used
// to leave these offering letters that no longer exist and hiding the one that does --
// which is how a placement row became impossible to point at the piece it meant.
function refreshPieceMenus() {
  const form = document.getElementById('form');
  if (!form) return;
  form.querySelectorAll('[data-cpath]').forEach(node => {
    if (node.dataset.cctl !== 'piece' && node.dataset.cctl !== 'pieces') return;
    const root = node.dataset.croot, path = JSON.parse(node.dataset.cpath);
    const field = fieldAt(root, path), target = rootTarget(root);
    if (!field || !target) return;
    // A branch of an `either` -- `extinction.pieces` is one -- keeps its switcher
    // outside the field, so the whole thing is redrawn or the switcher goes with it.
    const box = field.control === 'either' ? node.closest('.either') : node.closest('.field');
    if (box) box.outerHTML = fieldHtml(field, getIn(target, path), path, root, []);
  });
  form.querySelectorAll('[data-pfield="piece"]').forEach(node => {
    const row = ((DRAFT.setup || {}).placements || [])[Number(node.dataset.place)] || {};
    node.innerHTML = (DRAFT.pieces || []).map(piece =>
      `<option value="${esc(piece.id)}"${piece.id === row.piece ? ' selected' : ''}
        >${esc(piece.id)}</option>`).join('');
  });
  wireCards(form);
  wirePlacements(form);
}

// A way of moving's rows, wherever they are drawn: a piece's card, or a portal group's
// (owner `p<n>`, spelling 31, Q1007b), which the portal cards draw again on their own.
function wireMoveRows(root) {
  if (!root || !root.querySelectorAll) return;
  const changed = structural => {
    if (structural) drawDesigner();
    checkDraft();
  };
  root.querySelectorAll('[data-move]').forEach(node => {
    const [owner, at] = node.dataset.move.split('.');
    const index = Number(at);
    const field = node.dataset.mfield;
    const write = settled => {
      const move = movesOwner(owner).moves[index];
      const before = [...moveExtrasSkip(move)].join();
      writeMoveField(move, field, node.value);
      if (!isPortalKey(owner)) DRAFT_PIECE = Number(owner);
      // A new kind of move can bring or take away a hop's fields (fix 8); the menu
      // that changed it is a select, so a redraw loses no cursor.
      if (field === 'kind' && [...moveExtrasSkip(move)].join() !== before) drawDesigner();
      // Q1007a: riding changes the kind and the range, which other controls show, and a new
      // kind, or directions typed and left (a change, not an input, so no cursor is lost),
      // may bring or take away the menu.
      else if (field === 'ride' || ((field === 'kind' || (field === 'vectors' && settled))
        && ridable(move)
        !== !!node.closest('.moveRow').querySelector('[data-mfield="ride"]'))) drawDesigner();
      checkDraft();
      loadMoveMap();
    };
    node.onchange = () => write(true);
    if (node.tagName === 'INPUT') node.oninput = () => write(false);
  });
  root.querySelectorAll('[data-addmove]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const owner = movesOwner(node.dataset.addmove);
      if (!owner) return;
      owner.moves = owner.moves || [];
      // A portal's new way of moving is the one step a portal may take (D1).
      owner.moves.push(isPortalKey(node.dataset.addmove) ? JSON.parse(JSON.stringify(PORTAL_STEP))
        : { kind: 'slide', vectors: [[0, 1]], symmetry: 'rotate4' });
      // On hexagons the slide it arrives with goes the hexagons' six ways (Q640).
      if ((DRAFT.board || {}).lattice === 'hex') return ontoHex();
      changed(true);
      loadMoveMap();
    };
  });
  root.querySelectorAll('[data-dropmove]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const [owner, index] = node.dataset.dropmove.split('.');
      const held = movesOwner(owner);
      held.moves.splice(Number(index), 1);
      // A group whose last way of moving goes writes none (spelling 31's one spelling).
      if (isPortalKey(owner) && !held.moves.length) delete held.moves;
      changed(true);
      loadMoveMap();
    };
  });
}

// The placement rows' own controls. Wired apart from the rest of the form because a
// changed letter redraws the piece menus without redrawing the form around them.
function wirePlacements(form) {
  form.querySelectorAll('[data-place]').forEach(node => {
    const index = Number(node.dataset.place), field = node.dataset.pfield;
    const write = () => {
      const placement = DRAFT.setup.placements[index];
      if (field === 'player') {
        if (Number(node.value) === 1) placement.player = 1; else delete placement.player;
      } else {
        placement[field] = field === 'squares'
          ? node.value.trim().split(/\s+/).filter(Boolean) : node.value;
      }
      checkDraft();
    };
    node.onchange = write;
    if (node.tagName === 'INPUT') node.oninput = write;
  });
}

function wireDesigner() {
  const form = document.getElementById('form');
  if (!form) return;

  const changed = (structural) => {
    if (structural) drawDesigner();
    checkDraft();
  };

  form.querySelector('#dName').oninput = e => { DRAFT.name = e.target.value; checkDraft(); };
  form.querySelector('#dDesc').oninput = e => {
    DRAFT.description = e.target.value; checkDraft();
  };
  form.querySelector('#dUndo').onclick = () => stepDraft(false);
  form.querySelector('#dRedo').onclick = () => stepDraft(true);
  drawUndo();
  form.querySelectorAll('[data-field]').forEach(node => {
    const index = Number(node.dataset.index), field = node.dataset.field;
    const write = () => {
      const piece = DRAFT.pieces[index];
      if (field === 'royal') {
        if (node.checked) piece.royal = true; else delete piece.royal;
        return changed(false);
      }
      if (field === 'value') { piece.value = Number(node.value) || 0; return changed(false); }
      if (field === 'icon') {
        DRAFT.appearance = DRAFT.appearance || {};
        DRAFT.appearance.icons = DRAFT.appearance.icons || {};
        // One name for both players, or one name each. A third is written as a third
        // and refused by the schema, which is where every judgement about a draft lives.
        const typed = node.value.trim().split(/\s+/).filter(Boolean);
        if (typed.length > 1) DRAFT.appearance.icons[piece.id] = typed;
        else if (typed.length) DRAFT.appearance.icons[piece.id] = typed[0];
        else delete DRAFT.appearance.icons[piece.id];
        if (!Object.keys(DRAFT.appearance.icons).length) delete DRAFT.appearance;
        return changed(false);
      }
      if (field === 'id') {
        // The letter is what every other rule calls this piece, so changing it here
        // changes it everywhere before anything else sees the draft. The menus that
        // list pieces are redrawn in place: redrawing the form would put the cursor
        // back at the start of this box on every letter typed.
        const from = piece.id;
        piece.id = node.value;
        if (PIECES_OPEN.delete(from)) PIECES_OPEN.add(node.value);
        renamePiece(from, node.value);
        refreshPieceMenus();
        return changed(false);
      }
      piece[field] = node.value;
      if (field === 'name') {
        const folded = form.querySelector(`[data-foldname="${index}"]`);
        if (folded) folded.textContent = node.value;
      }
      changed(false);
    };
    if (node.type === 'checkbox') node.onchange = write;
    else { node.oninput = write; node.onchange = write; }
  });

  form.querySelectorAll('[data-bfield]').forEach(node => {
    node.onchange = () => {
      DRAFT.board = DRAFT.board || {};
      const field = node.dataset.bfield;
      writeBoardField(DRAFT.board, field, node.value);
      // The lattice and the shape decide which size controls, which note and which wrap
      // rows are drawn, so either redraws the form.
      const reshaped = field === 'lattice' || field === 'shape';
      if (reshaped) fitWrapToShape();
      // Whatever this leaves off the board is taken away once the check says which it is.
      BOARD_RESHAPED = true;
      BOARD_REMOVED = '';
      // Hexagons refuse the square grid's readings (a move's symmetry, the mirrored army,
      // a rank zone), so the server rewrites each as its hex twin first; without that no
      // square game made hexagons was a game, and the board never changed (Q640). Asked
      // again when the shape changes, so a draft made hexagons before this still can be.
      if (reshaped && DRAFT.board.lattice === 'hex') return ontoHex();
      changed(reshaped);
    };
  });

  wireMoveRows(form);

  // Schema 30 (Q999c): each piece's lines of sight.
  wireSights(form, changed);

  wirePlacements(form);

  // A mirror derives the second army, and the schema refuses a `player` beside one, so
  // turning it on folds every row back to the first player.
  form.querySelector('#dMirror').onchange = e => {
    DRAFT.setup = DRAFT.setup || {};
    DRAFT.setup.mirror = e.target.checked;
    if (e.target.checked) (DRAFT.setup.placements || []).forEach(p => { delete p.player; });
    PLACING = null;
    changed(true);
  };
  form.querySelector('#dOrientation').onchange = e => {
    DRAFT.setup = DRAFT.setup || {};
    if (e.target.value === 'rotate') DRAFT.setup.orientation = 'rotate';
    else delete DRAFT.setup.orientation;
    changed(false);
  };
  form.querySelector('#dAddPlacement').onclick = () => {
    DRAFT.setup = DRAFT.setup || { placements: [] };
    DRAFT.setup.placements = DRAFT.setup.placements || [];
    DRAFT.setup.placements.push({ piece: (DRAFT.pieces[0] || {}).id || '', squares: [] });
    changed(true);
  };
  form.querySelectorAll('[data-dropplace]').forEach(node => {
    node.onclick = () => {
      DRAFT.setup.placements.splice(Number(node.dataset.dropplace), 1);
      changed(true);
    };
  });

  wireCards(form);

  // A piece with no way of moving is a validation error, and one with no placement never
  // reaches the board -- so a new piece arrives with both, and the editor's next check is
  // about what you meant to change rather than about what you have not filled in yet.
  form.querySelector('#dAddPiece').onclick = () => {
    const used = new Set((DRAFT.pieces || []).map(p => p.id));
    const letter = 'ABCDEFGHJLMTUVWXYZ'.split('').find(c => !used.has(c)) || 'Z';
    DRAFT.pieces.push({
      id: letter, name: 'new piece', value: 1,
      moves: [{ kind: 'leap', vectors: [[1, 2]], symmetry: 'dihedral8' }],
    });
    DRAFT_PIECE = DRAFT.pieces.length - 1;
    freshPiece(DRAFT_PIECE);
    // On hexagons the knight's leap it arrives with is the hex knight's (Q640).
    if ((DRAFT.board || {}).lattice === 'hex') return ontoHex();
    changed(true);
    loadMoveMap();
  };
  form.querySelectorAll('[data-drop]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const index = Number(node.dataset.drop);
      const id = DRAFT.pieces[index].id;
      DRAFT.pieces.splice(index, 1);
      // A placement naming a piece that no longer exists is a validation error, and
      // leaving one behind would report the deletion as a broken draft.
      if (DRAFT.setup && DRAFT.setup.placements) {
        DRAFT.setup.placements = DRAFT.setup.placements.filter(p => p.piece !== id);
      }
      DRAFT_PIECE = Math.min(DRAFT_PIECE, Math.max(0, DRAFT.pieces.length - 1));
      changed(true);
      loadMoveMap();
    };
  });
  // Every click inside a card lands here, including one that focuses its inputs. A
  // redraw would replace the focused input and lose the cursor, so the highlight
  // moves in place instead.
  form.querySelectorAll('[data-piece]').forEach(node => {
    node.onclick = () => {
      const piece = Number(node.dataset.piece);
      if (piece === DRAFT_PIECE) return;
      DRAFT_PIECE = piece;
      form.querySelectorAll('[data-piece]').forEach(card => {
        const on = Number(card.dataset.piece) === piece;
        card.classList.toggle('on', on);
      });
      loadMoveMap();
    };
  });
  wireUploads(form);
  wirePicker(form);
  wireReadyNeeds(form);
  wireFolds(form);
  wirePortalCards(form);

  // Read by the server as a rule file is (Q830b): the parse, the size, the validator and
  // the compiler are its, and a text with a mistake leaves the draft as it was.
  form.querySelector('#dJsonApply').onclick = () => loadRuleText(
    form.querySelector('#dJson').value, { fresh: false, near: 'dPasteSaid', named: '' });
}

// An SVG the reader has on disk, inlined into the spec. An icon entry is free text, so a
// drawing needs no schema change, travels with the ruleset, and stays outside the content
// hash where appearance belongs -- a game does not become a different game because its
// knight got a new picture.
//
// `seat` is which player the drawing is for, or `null` where one drawing serves both.
function uploadIcon(index, seat) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.svg,image/svg+xml';
  input.onchange = () => {
    const file = input.files && input.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      api('/api/design/icon', { svg: String(reader.result) }).then(answer => {
        const piece = DRAFT.pieces[index];
        DRAFT.appearance = DRAFT.appearance || {};
        DRAFT.appearance.icons = DRAFT.appearance.icons || {};
        const held = DRAFT.appearance.icons[piece.id];
        DRAFT.appearance.icons[piece.id] = seat === null
          ? answer.href : seatPair(held, piece, seat, answer.href);
        drawDesigner();
        checkDraft();
      }).catch(fail);
    };
    reader.readAsText(file);
  };
  input.click();
}

// One seat's drawing written into the pair, leaving the other seat alone. A piece that
// held one entry, or none, grows into a pair seeded with what the untouched seat is
// already wearing -- its typed name, or the piece's own name, which is what the icon
// lookup falls back to -- so that seat keeps the picture it had.
function seatPair(held, piece, seat, href) {
  const other = String(held || piece.name || piece.id || '');
  const pair = Array.isArray(held) && held.length === 2 ? held.slice() : [other, other];
  pair[seat] = href;
  return pair;
}

// Asked after a pause rather than on every keystroke: the check compiles the spec and
// plays a hundred and fifty random games, which is a fifth of a second and not something
// to do per letter typed into a description.
function checkDraft() {
  if (DRAFT_TIMER) clearTimeout(DRAFT_TIMER);
  DRAFT_TIMER = setTimeout(() => {
    snapshotDraft();
    const asked = JSON.stringify(DRAFT);
    // With the game it was forked from, so the answer can say what changed from it.
    api('/api/design/check', { spec: DRAFT, parent: DRAFT_FROM }).then(answer => {
      const current = JSON.stringify(DRAFT) === asked;
      DRAFT_CHECK = answer;
      if (fitAfterReshape(answer)) return;
      if (answer.preview) DRAFT_PREVIEW = answer.preview;
      // When the answer came, said on the verdict so a re-check is seen to have happened
      // even when its count did not move (F10: *11 opening moves* before and after two
      // knights went down, and nothing showed the second check had run).
      DRAFT_CHECKED = new Date();
      if (current) {
        MOVE_SAID = { key: JSON.stringify(DRAFT), lines: answer.moveLines || [],
                      portalLines: answer.portalMoveLines || [] };
        drawMoveSays();
        drawFoldSays();
        SIGHT_SAID = { key: JSON.stringify(DRAFT), lines: answer.sightLines || [],
                       problems: answer.sightProblems || [] };
        drawSightSays();
      }
      if (answer.valid && answer.hash !== DRAFT_HASH) {
        DRAFT_HASH = answer.hash;
        DRAFT_STANDING = null;
        DRAFT_SCREEN = null;
        DRAFT_SAVED = '';
        loadStanding(answer.hash);
        runScreen(answer.hash);
      }
      drawVerdict();
      // Whether the draft still has the old rest switch, whose change is offered beside
      // *every piece rests* (Q995d).
      if (current) drawRestAll();
      // The icons come back with the check, so a piece that just got a drawing shows it.
      const slots = document.querySelectorAll('[data-icon]');
      slots.forEach(slot => { slot.innerHTML = pieceIconHtml(Number(slot.dataset.icon)); });
      // And so does whether it is drawn per player, which is what decides between one
      // upload button and one per seat. Refreshed here rather than by redrawing the
      // form, because a piece turns paired while its icon field is being typed into.
      document.querySelectorAll('[data-uploads]').forEach(box => {
        box.innerHTML = uploadButtonsHtml(Number(box.dataset.uploads));
        wireUploads(box);
      });
      refreshPalette();
      // The pairs' sentences and warnings, the refusals that name them, and whether
      // another pair can go in (P3).
      drawPortalSays();
      // The duck's rule in words and its refusals (generation 42, N4c).
      drawDuckSays();
      // Which rules cannot go in this game, asked once the draft is one. A card reads
      // the opening position for the piece it needs, so this waits for the check.
      if (answer.valid) loadRefusals();
      // And which design steps apply to it now. Asked after every change, because a
      // step is drawn against the draft as it stands and a list from two edits ago is
      // a list of other games' changes.
      refreshTry();
    }).catch(error => {
      DRAFT_CHECK = { valid: false, errors: [errorWords(error)] };
      drawVerdict();
    });
  }, 320);
}

// ------------------------------------------------------- taking a change back

// The draft as it stands, recorded if it has moved since the last one. Called from the
// check, which is debounced, so a burst of typing is one step back rather than one per
// letter -- and every change reaches the check, so nothing can change without passing
// here first.
function snapshotDraft() {
  const now = JSON.stringify(DRAFT);
  // What a design step changed is true of the draft that press made and of no other, so
  // an edit or an undo takes the sentence with it.
  if (TRY_SAID && TRY_SAID.key !== now) TRY_SAID = null;
  // So is what a turn-taking choice changed (Q995b).
  if (TURN_SAID && TURN_SAID.key !== now) TURN_SAID = null;
  // And so is a game in hand: it was played under the rules as they stood when it
  // started, and these are not those rules any more.
  if (BENCH_PLAY && BENCH_PLAY.key !== now) {
    endBenchGame('The rules changed while that game was going, so it is over. It was a'
      + ' game of the rules as they stood when it started. Start another whenever you'
      + ' like.');
  }
  if (now === DRAFT_SNAP) return;
  if (DRAFT_SNAP) {
    DRAFT_UNDO.push(DRAFT_SNAP);
    if (DRAFT_UNDO.length > UNDO_DEPTH) DRAFT_UNDO.shift();
    // A new change is a new branch, so what was undone is no longer ahead of anything.
    DRAFT_REDO = [];
  }
  DRAFT_SNAP = now;
  keepDraft();
  drawUndo();
}

// One step back, or forward. The whole draft is replaced, so a rule half added and the
// piece it named come back together -- which a per-field undo could not promise.
function stepDraft(forward) {
  // Undo while a pair is being put down puts the placement away and nothing else (D8).
  if (!forward && cancelPortalTool()) return;
  const from = forward ? DRAFT_REDO : DRAFT_UNDO;
  const to = forward ? DRAFT_UNDO : DRAFT_REDO;
  if (!from.length) return;
  to.push(JSON.stringify(DRAFT));
  const before = DRAFT;
  DRAFT = JSON.parse(from.pop());
  resetPortalState();
  // What came back, where it was a portal (D8): *Pair 2 is back, on b4 and e3*.
  benchSay(portalUndoWords(before, DRAFT));
  DRAFT_SNAP = JSON.stringify(DRAFT);
  DRAFT_PIECE = Math.min(DRAFT_PIECE, Math.max(0, (DRAFT.pieces || []).length - 1));
  // Each of these points at something in the draft that has just been replaced.
  PICKING = null;
  PLACING = null;
  TEACHING = null;
  keepDraft();
  drawDesigner();
  drawUndo();
  checkDraft();
  loadMoveMap();
}

function drawUndo() {
  const undo = document.getElementById('dUndo'), redo = document.getElementById('dRedo');
  if (undo) undo.disabled = !DRAFT_UNDO.length;
  if (redo) redo.disabled = !DRAFT_REDO.length;
}

// The draft, kept where a reload can find it. Storage can be absent, full or refused,
// and none of those is a reason for the editor to stop working.
function keepDraft() {
  try {
    // The spellings a rule card is showing ride along (Q541): one taken and not yet
    // filled in writes no key into the draft, so the draft alone cannot say it.
    window.sessionStorage.setItem(DRAFT_KEY,
      JSON.stringify({ spec: DRAFT, from: DRAFT_FROM, spellings: CHOSEN_SPELLING }));
  } catch (error) { /* the draft lives in the page instead */ }
}

function heldDraft() {
  try {
    const held = JSON.parse(window.sessionStorage.getItem(DRAFT_KEY) || 'null');
    return held && held.spec && typeof held.spec === 'object' ? held : null;
  } catch (error) { return null; }
}

// The draft a reload left behind, or `template` when there is none, taken up as the one
// being edited: its spec, the game it was started from, and the spellings its rule cards
// were showing.
// A template is a starting point, and its sentence describes the template. A draft saved
// with it kept that sentence, so the library said the designer's game was the template
// (the first-visit study's F9): the description starts empty, and the field asks for one.
// A fork already starts empty (`forkOf`), and a draft held over a reload keeps
// whatever its designer wrote.
// Starting from an existing game: a copy of its rules, under a new name, because the
// thing about to happen to it is being changed. Keeping the old name would leave two
// different games called chess in one library, and so would a name some other library
// game already has, so the name counts up past those. The game it came from is never
// touched: the copy is made of the answer, and the answer is not kept.
// The parent's own provenance goes (it names *its* parents and authors, which this draft
// is not), and the one parent this was copied from is said in the schema's own words,
// `provenance.parents`, which is outside the mechanics hash (Q642).
function forkOf(source, from) {
  const draft = JSON.parse(JSON.stringify(source.spec));
  draft.name = freshName((source.name || draft.name || 'a game') + ' variant');
  draft.description = '';
  draft.provenance = { parents: [from] };
  return draft;
}

function freshName(wanted) {
  const taken = new Set(LIBRARY.map(game => String(game.name || '').toLowerCase()));
  let name = wanted;
  for (let count = 2; taken.has(name.toLowerCase()); count += 1) name = `${wanted} ${count}`;
  return name;
}

// The draft's `provenance.parents` says what `DRAFT_FROM` says, at the save, which is the
// one place it is read. Undo restores whole drafts and not the game they started from,
// so a copy's `parents` can outlive a switch to a template; the save puts that right.
// Anything else a person wrote into `provenance` stays.
// The chooser, *start again from*, offers a library game too (Q642). The templates stay first and stay the
// default; every game the library lists follows, by name, under a prefix no template key
// carries. Nothing is listed when the library has not answered.
const GAME_CHOICE = 'game:';
let AFRESH = false;

function startingGamesHtml() {
  const games = LIBRARY.filter(game => game && game.hash && game.name)
    .slice().sort((a, b) => String(a.name).localeCompare(String(b.name)));
  if (!games.length) return '';
  return `<optgroup label="games in the library">${games.map(game =>
    `<option value="${esc(GAME_CHOICE + game.hash)}">${esc(game.name)}</option>`).join('')
  }</optgroup>`;
}

// The chooser stands above the board rather than in the form (F10, `first-visit-study.md`):
// on a phone the form comes after the board, and a reader looking for somewhere to start
// met it some two screens down. Drawn once with the page, so it keeps no state of its own:
// a choice is acted on and the box goes back to its question.
function startFromHtml(kit) {
  return `<label class="field startFrom"><span>start from a template, or a copy of a game
      in the library</span>
    <select id="dTemplate">
      <option value="">choose one...</option>
      <optgroup label="templates">${(kit.templates || []).map(t =>
        `<option value="${esc(t.key)}" title="${esc(t.means)}">${esc(t.title)}</option>`
      ).join('')}</optgroup>
      ${startingGamesHtml()}
    </select></label>
    ${ruleFileHtml(kit)}`;
}

// ------------------------------------------------------------- a rule file, loaded
//
// Q830b (`plan-website.md`, *LLM co-design*): the site offers no AI. A person who wants
// one writes the game outside, by hand or with their own, from the rule language as one
// file in the public repository, and brings the file here. The links (the page, G8, and the
// Markdown file for an AI), the button's label and how much of a file to send are the kit's
// (`designing.rule_file_kit`), so each link is spelt once and `RULES.md`'s *Trying a game* names this button by the same string.
//
// **The page reads nothing in the file.** It sends the text to `POST /api/design/load`,
// which parses it, holds it to its size and depth, and asks the validator and the compiler
// the designer's check asks. A valid file comes back as the draft; an invalid one comes
// back as the validator's own lines, shown here, and the draft in hand is not touched.
// The paste box at the foot of the form (*the game as text*) goes through the same route.

function ruleFileHtml(kit) {
  const file = kit && kit.ruleFile;
  if (!file) return '';
  return `<div class="ruleFile">
      <div class="row small">
        <button id="dLoadFile" type="button">${esc(file.load)}</button>
        <span class="muted">Writing a game by hand, or with your own AI? Read
          <a href="${esc(file.rulesPage || file.rulesUrl)}" target="_blank" rel="noopener"
            title="opens in a new tab">the rule language</a> (for an AI,
          <a href="${esc(file.rulesUrl)}" target="_blank" rel="noopener"
            title="opens in a new tab">the same as one Markdown file</a>), write the game as a
          <code>.json</code> file, and load it here.</span>
      </div>
      <div id="dLoadSaid" class="ruleSaid" aria-live="polite"></div>
    </div>`;
}

// The paste box's name, which *Trying a game* names too.
function pasteLabel() {
  return ((DESIGN_KIT || {}).ruleFile || {}).paste || 'the game as text';
}

function wireRuleFile() {
  const button = document.getElementById('dLoadFile');
  if (!button) return;
  button.onclick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,application/json';
    input.onchange = () => {
      const file = input.files && input.files[0];
      if (file) loadRuleFile(file);
    };
    input.click();
  };
}

// One byte past the limit is all the server needs to refuse a file for its size, so a
// large file is neither read whole nor sent whole.
function loadRuleFile(file) {
  const limit = Number(((DESIGN_KIT || {}).ruleFile || {}).bytes) || 0;
  const part = limit && file.size > limit ? file.slice(0, limit + 1) : file;
  return Promise.resolve(part.text()).then(text => loadRuleText(text, {
    fresh: true, near: 'dLoadSaid', named: String(file.name || 'that file'),
  })).catch(error => ruleFileSaid('dLoadSaid', [errorWords(error)], ''));
}

// The text of a file or of the paste box, sent to be read. `fresh` is a file: a game of
// its own, started from nothing in the library. A paste keeps the game this draft was
// started from, since the box holds this draft's text to be edited.
function loadRuleText(text, how) {
  return api('/api/design/load', { text: String(text) }).then(answer => {
    if (!answer.valid) {
      ruleFileSaid(how.near, answer.errors || [], how.named);
      return answer;
    }
    takeLoadedSpec(answer.spec, how);
    return answer;
  }).catch(error => ruleFileSaid(how.near, [errorWords(error)], how.named));
}

// A file or a paste that was a game becomes the draft.
function takeLoadedSpec(spec, how) {
  DRAFT = spec;
  resetPortalState();
  FRESH_PIECE = null;
  if (how.fresh) {
    DRAFT_FROM = '';
    DRAFT_PARENT = null;
  }
  DRAFT_PIECE = 0;
  PICKING = null;
  PLACING = null;
  TEACHING = null;
  drawDesigner();
  checkDraft();
  loadMoveMap();
  ruleFileSaid(how.near, [], how.named);
}

// What became of it, under the control that sent it: the validator's lines as they came,
// or that the draft is now this game, which undo takes back. A file in an older spelling
// is read in the current one by the server (`spelling.current_spelling`), so it loads.
function ruleFileSaid(near, errors, named) {
  const slot = document.getElementById(near);
  if (!slot) return;
  const what = named ? `<b>${esc(named)}</b>` : 'That text';
  slot.innerHTML = errors.length
    ? `<div class="error small" role="alert">
        <div>${what} was not loaded, and the game you were editing is as it was:</div>
        <ul class="rules">${errors.map(line => `<li><code>${esc(line)}</code></li>`).join('')}</ul>
      </div>`
    : `<div class="ok small">${what} is your draft now: <b>${esc(DRAFT.name || 'a game')}</b>.
        Undo takes it back, and nothing is kept until you save it.</div>`;
}

function wireStartFrom() {
  const chooser = document.getElementById('dTemplate');
  if (!chooser) return;
  // No confirmation: undo covers it, and a dialog in front of a template was one more
  // thing to get past on the way to trying something.
  chooser.onchange = e => {
    // A library game is opened the way its page's *make your own from this* opens it, so
    // the copy, its new name and *Started from* above the form are made in one place.
    const game = String(e.target.value || '');
    e.target.value = '';
    if (game.startsWith(GAME_CHOICE)) {
      const hash = game.slice(GAME_CHOICE.length);
      // Chosen again, the game this draft is already a copy of starts the copy afresh.
      AFRESH = true;
      if (hash) go('#/design/' + hash);
      return;
    }
    const found = (DESIGN_KIT.templates || []).find(t => t.key === game);
    if (!found) return;
    DRAFT = fromTemplate(found.spec);
    resetPortalState();
    FRESH_PIECE = null;
    DRAFT_FROM = '';
    DRAFT_PARENT = null;
    DRAFT_PIECE = 0;
    drawDesigner();
    checkDraft();
    loadMoveMap();
  };
}

function stampParent() {
  const kept = { ...(DRAFT.provenance || {}) };
  if (DRAFT_FROM) kept.parents = [DRAFT_FROM]; else delete kept.parents;
  if (Object.keys(kept).length) DRAFT.provenance = kept; else delete DRAFT.provenance;
}

function fromTemplate(spec) {
  const draft = JSON.parse(JSON.stringify(spec));
  draft.description = '';
  return draft;
}

function takeDraft(held, template) {
  DRAFT = held ? JSON.parse(JSON.stringify(held.spec)) : fromTemplate(template);
  DRAFT_FROM = held ? (held.from || '') : '';
  DRAFT_PARENT = null;
  const spellings = held && held.spellings;
  CHOSEN_SPELLING = spellings && typeof spellings === 'object' && !Array.isArray(spellings)
    ? spellings : {};
}

// Ctrl+Z and Ctrl+Y, wired once for the page and inert unless the editor is on screen.
// They take the *draft* back, not the field: a person who has just removed a piece is
// not asking for the last letter they typed.
function wireUndoKeys() {
  if (wireUndoKeys.done) return;
  wireUndoKeys.done = true;
  document.addEventListener('keydown', event => {
    const key = (event.key || '').toLowerCase();
    if (key !== 'z' && key !== 'y') return;
    if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
    if (!document.getElementById('form')) return;
    event.preventDefault();
    stepDraft(key === 'y' || event.shiftKey);
  });
}

// The opening position the editor draws on, which is the last one a check could draw.
// A half-made edit leaves the draft invalid and gives it no board of its own, and the
// board is usually the tool the repair needs -- so the palette, the square picker and
// the side column keep working on this one until a check draws the next.
function lastPreview() {
  return ((DRAFT_CHECK || {}).preview) || DRAFT_PREVIEW || null;
}

// What the engine says about the chosen piece, for whichever board is on screen. *Where
// it goes* asks about the opening position as it stands; the other two ask what the
// descriptors mean, which is a fact about an empty board.
function loadMoveMap() {
  DRAFT_MAP = null;
  const piece = DRAFT_PIECE, mode = DRAFT_MODE;
  // A drawing belongs to one piece standing on one square.
  if (TEACHING && TEACHING.piece !== piece) TEACHING = null;
  api('/api/design/moves',
      { spec: DRAFT, piece, square: DRAFT_SQUARE, alone: mode !== 'goes' })
    .then(map => {
      if (DRAFT_PIECE !== piece || DRAFT_MODE !== mode) return;
      DRAFT_MAP = map;
      drawVerdict();
    })
    .catch(() => { DRAFT_MAP = null; });
}

// ------------------------------------------------------------- and now try
//
// Options C1 and C2. The invention loop knows a few dozen mechanical edits, and until now
// the only way to press one was from the map, on a game the library already holds. This
// offers the same edits against the draft in hand -- which is usually nobody's game, has
// no hash anybody could name, and may not be a game at all yet.
//
// **What can be offered is the server's answer.** `POST /api/design/steps` draws every
// operator against this draft under fixed seeds and reports the ones that land, each with
// the operator's own sentence; the page prints what it is given and asks again whenever
// the draft has moved. It decides nothing about which edits exist or which apply.
//
// A press is a change to the draft like any other change here: it goes through the same
// check, so the same snapshot records it and Ctrl+Z takes it back.

let TRY_OPEN = false,  // whether the panel has been asked for
    TRY = null,        // {key, state, steps, why, failed, making} for the draft asked about
    TRY_SAID = null;   // {key, words, from} -- the last press, while the draft is what it made

function tryPanelHtml() {
  return `<div class="panel u-mt-14">
    <h3 class="u-mt-0">And now try&hellip;</h3>
    <div id="dTry">${tryHtml()}</div>
  </div>`;
}

function tryHtml() {
  if (!TRY_OPEN) {
    return `<div class="muted small">Small changes the site knows how to make, each one
        tried on this draft. One press makes the change, leaves a draft you
        can play, and says what it did; undo takes it back and nothing is kept until you
        save.</div>
      <button id="dTryAsk" class="u-mt-8">and now try&hellip;</button>`;
  }
  const held = TRY || { state: 'asking' };
  // What the last press changed stands above whatever the panel is showing now, because
  // the first thing that happens after a press is the list being asked for again.
  const said = trySaidHtml();
  if (held.state === 'asking') {
    return `${said}<div class="muted small">Looking for the changes that apply to this
      draft&hellip; about a twentieth of a second.</div>`;
  }
  // A request that went wrong, or a press the server refused. Said under whatever the
  // panel was already showing rather than in place of it, so the list stays pressable.
  const failed = held.failed
    ? `<div class="muted small u-mt-8">${esc(held.failed)}</div>` : '';
  if (held.why) {
    // The draft is not a game yet. This is the validator's own sentence, and the verdict
    // beside it is showing the same one.
    return `${said}<div class="muted small">No change is offered while these rules are
      not a game. ${esc(held.why)}</div>${failed}`;
  }
  if (!held.steps.length) {
    return `${said}<div class="muted small">No change the program knows makes a different
      game out of this draft.</div>${failed}`;
  }
  return `${said}
    <div class="muted small">${held.steps.length} changes each make a different game out
      of this draft. Each is one press, and undo takes it back.${held.back
      ? ' A change that would only take back the last one is left out: that is undo.'
      : ''}${
      held.making ? ' <b>Making it&hellip;</b>' : ''}</div>
    <div class="tryList">${held.steps.map(step => `
      <button data-trystep="${esc(step.operator)}" data-seed="${esc(String(step.seed))}"
        ${held.making ? 'disabled' : ''}>${esc(step.words)}</button>`).join('')}</div>
    ${failed}`;
}

// What the last press changed, in the operator's own words, for as long as the draft is
// the one that press made. `snapshotDraft` drops it the moment anything else moves.
function trySaidHtml() {
  if (!TRY_SAID) return '';
  return `<div class="ok small u-mt-0 u-mb-8">${
    esc(TRY_SAID.words)}. Undo takes it back.</div>`;
}

function drawTry() {
  const box = document.getElementById('dTry');
  if (!box) return;
  box.innerHTML = tryHtml();
  wireTry();
}

// Found through the document rather than through a root, because the panel is redrawn on
// its own and by the whole verdict column, and both would have to pass the same root.
// `data-trystep` is this panel's alone; the map's own step rows carry `data-step`.
function wireTry() {
  const ask = document.getElementById('dTryAsk');
  if (ask) {
    ask.onclick = () => { TRY_OPEN = true; TRY = null; refreshTry(); };
  }
  document.querySelectorAll('[data-trystep]').forEach(node => {
    node.onclick = () => takeStep(node.dataset.trystep, Number(node.dataset.seed));
  });
}

// The list, for the draft as it stands. Asked once per draft: the check is debounced and
// calls this after it, so a burst of typing is one request rather than one per letter.
function refreshTry() {
  if (!TRY_OPEN) return;
  const key = JSON.stringify(DRAFT);
  if (TRY && TRY.key === key) return;
  TRY = { key, state: 'asking', steps: [], why: '', failed: '', making: '' };
  drawTry();
  // Right after a press, the draft it was made on goes too, and a step that would land
  // back there is not offered: several steps toggle a rule, and pressing one twice walked
  // the draft back to where it started (Q532). Undo is that step, and says so.
  const back = TRY_SAID && TRY_SAID.key === key && TRY_SAID.from
    ? JSON.parse(TRY_SAID.from) : null;
  const asked = back ? { spec: DRAFT, back } : { spec: DRAFT };
  return api('/api/design/steps', asked).then(answer => {
    if (JSON.stringify(DRAFT) !== key) return;
    TRY = { key, state: 'ready', steps: answer.steps || [], why: answer.why || '',
            failed: '', making: '', back: !!back };
    drawTry();
  }).catch(error => {
    if (JSON.stringify(DRAFT) !== key) return;
    TRY = { key, state: 'ready', steps: [], why: '', making: '',
            failed: errorWords(error) };
    drawTry();
  });
}

// One step, pressed. The same draw the list showed, under the seed it named, so what is
// on the button is what arrives -- and the draft that comes back is the whole answer,
// because the server has already compiled it.
function takeStep(operator, seed) {
  if (!TRY || TRY.making) return;
  TRY.making = operator;
  drawTry();
  const from = JSON.stringify(DRAFT);
  return api('/api/design/steps', { spec: DRAFT, step: operator, seed }).then(answer => {
    const made = (answer || {}).remix;
    if (!made || !made.spec) throw new Error('that step made nothing of this draft');
    DRAFT = made.spec;
    resetPortalState();
    DRAFT_PIECE = 0;
    DRAFT_SQUARE = -1;
    FRESH_PIECE = null;
    // Each of these points at something in the draft that has just been replaced.
    PICKING = null;
    PLACING = null;
    TEACHING = null;
    TRY_SAID = { key: JSON.stringify(DRAFT), words: made.words, from };
    TRY = null;
    drawDesigner();
    drawTry();
    // The change is recorded by the check's own snapshot, which is what puts it behind
    // undo, and the list is asked again for the draft it left.
    checkDraft();
    loadMoveMap();
  }).catch(error => {
    if (TRY) { TRY.making = ''; TRY.failed = errorWords(error); }
    drawTry();
  });
}

// ------------------------------------------------------------- what the draft is

function drawVerdict() {
  drawBoardSide();
  const side = document.getElementById('verdictSide');
  if (!side) return;
  const check = DRAFT_CHECK;
  if (!check) {
    side.innerHTML = '<div class="panel muted">checking...</div>';
    reserveFootbar();
    return;
  }

  if (!check.valid) {
    // The board beside this is the draft's own, drawn from a stand-in when the check
    // could (Q640), else the last position a check could draw. Most broken drafts are an
    // edit somebody has not finished, the way back is usually a piece put down or a
    // square picked, and removing the board took that away at the moment it was needed.
    const own = !!(check.preview && check.preview.standIn);
    const errors = check.errors || [];
    // Compact, because it complains about a setting the person is in the middle of
    // changing (Q580; the phone pass, `first-visit-study.md` §1.2a: *"the panel takes up
    // half the space, making it harder to fix what it complains about"*). One line says
    // so with the first thing to fix; the rest, and why the board still takes pieces, are
    // one tap away. Open by default where the verdict has a column of its own.
    side.innerHTML = `<div class="panel">
      <div class="footbar" id="dActions">
        <div class="says">
          <h3 class="notGame">Not a game yet</h3>
          <div class="notGameFirst small">${esc(errors[0] || '')}</div>
          <details class="notGameAll" id="dNotGame">
            <summary class="muted small">${errors.length > 1
              ? `all ${errors.length} things to fix` : 'more about this'}</summary>
            <div class="muted small">The rules have to make sense before anything can play
              them, and these do not yet. ${own ? 'The board is this draft&rsquo;s own'
                : 'The board is the one these rules last made'}, and putting a piece down on
              it still works.</div>
            <ul class="rules">${errors.map(e => `<li>${esc(e)}</li>`).join('')}</ul>
          </details>
        </div>
      </div>
    </div>`;
    const all = document.getElementById('dNotGame');
    if (all) {
      all.open = NOT_GAME_OPEN === null ? !footbarFixed() : NOT_GAME_OPEN;
      all.ontoggle = () => { NOT_GAME_OPEN = all.open; footbarMoved(); };
    }
    reserveFootbar();
    keepEditInView();
    return;
  }

  const gate = check.playable === false
    ? `<div class="error">
         <b>It plays, but it does not finish.</b> ${checkedHtml()}
         <ul class="u-m-6-0-0-18 u-p-0">${
           check.playableReasons.map(r => `<li>${esc(r)}</li>`).join('')}</ul>
         <div class="muted small u-mt-6">You can still save it and play
           it yourself. This is a hundred and fifty games of random moves finding no
           winner, which usually means the ending is out of reach.</div>
         <button id="dStuck" class="u-mt-8">open a game that did not finish</button>
       </div>`
    : `<div class="ok">It plays and reaches an ending, with
         ${esc(check.openingWords || check.openingMoves + ' opening moves')}. ${checkedHtml()}</div>`;

  side.innerHTML = `
    <div class="panel">
      <h3 class="u-mt-0">${esc(DRAFT.name || 'this game')}</h3>
      <div class="footbar" id="dActions">
        <div class="says">${gate}</div>
        <div class="row">
          <button class="primary" id="dSave">save it</button>
          <span class="awayLead muted small">${esc(AWAY_TAIL.trim())}:</span>
          <button id="dPlay" aria-label="${esc(PLAY_AWAY)}"
            title="the full board, with the move list; nothing is saved"
            >${awayLabel(PLAY_AWAY)}</button>
          <button id="dWatch" aria-label="${esc(WATCH_AWAY)}" title="${esc(watchTitle(watchesRounds()))}"
            >${awayLabel(WATCH_AWAY)}</button>
        </div>
      </div>
      <div class="muted small u-mt-6">
        Playing it keeps nothing. It is a game against the rules as they stand right now,
        on the board page with its move list, and closing it leaves no trace; the
        <i>play it here</i> tab plays it beside the editor instead. Watching it, both sides are played by a computer
        player ${watchesRounds() ? 'for games in rounds, the strongest quick enough to watch'
        : `that looks at ${watchNodes()} positions before each move`}, and you can pause
        and step back through every one of them. ${ACCOUNTS
          ? 'A saved game is kept on this site as your draft.'
          : `A saved game is kept with the ones a design run invented${info('candidate')}.`}${
          offered('POST', '/api/design/measure')
          ? ' Below, you can measure it and nominate it.' : ''}${offered('POST', '/api/run')
          ? ' From its page you can start a design run from it.' : ''}
      </div>
      <div id="dSaved">${DRAFT_SAVED}</div>
      <div id="dLadder" class="ruled"
        >${ladderHtml(check)}</div>
      <div id="dHereAway">${hereAwayHtml(check)}</div>
      <div id="dFamily" class="ruled"
        >${familyHtml()}</div>
    </div>
    ${identityHtml(check)}
    ${changesHtml(check)}
    <div class="panel u-mt-14">
      <h3 class="u-mt-0">The rules, in English</h3>
      <ul class="rules small">${
        (check.ruleLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      <details class="u-mt-8">
        <summary class="muted small">how each piece moves</summary>
        <ul class="rules muted small u-mt-6">${
          (check.pieceLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      </details>
      <details class="u-mt-8">
        <summary class="muted small">where the pieces start</summary>
        <ul class="rules muted small u-mt-6">${
          (check.setupLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      </details>
    </div>`;

  document.getElementById('dSave').onclick = () => saveDraft();
  wireNameAsk(document.getElementById('dSaved'));
  document.getElementById('dPlay').onclick = () => playDraft(['human', PLAY_AGENT]);
  document.getElementById('dWatch').onclick = () =>
    playDraft([WATCH_AGENT, WATCH_AGENT]);
  const stuck = document.getElementById('dStuck');
  if (stuck) {
    stuck.onclick = () => showDraft('unfinished', 'Playing random games to find one that did not finish…');
  }
  wireLadder();
  wireHere();
  reserveFootbar();
}

// Whether the person opened or closed the full list of what is wrong; null until they
// have, when it follows the layout (closed in the bar at the foot, open in a column).
let NOT_GAME_OPEN = null;

// The setting just changed stays in view (Q580). On a phone the verdict is a bar fixed to
// the foot of the screen, and when the change makes the draft no game the bar grows by
// the sentence that says so, over the very control that was changed. The page scrolls by
// what the bar covers of it, and no further.
function keepEditInView() {
  if (!footbarFixed()) return;
  const bar = document.getElementById('dActions');
  const form = document.getElementById('form');
  const edited = document.activeElement;
  if (!bar || !form || !edited || !form.contains(edited)) return;
  const top = bar.getBoundingClientRect().top;
  const under = edited.getBoundingClientRect().bottom + 8 - top;
  if (under > 0) window.scrollBy(0, under);
}

function footbarMoved() {
  reserveFootbar();
  keepEditInView();
}

// When the last check answered, to the second: the count beside it can come back the
// same after an edit (two knights put down where the king and the rooks had been going),
// and the time is what says the edit was checked at all (F10).
function checkedHtml() {
  if (!DRAFT_CHECKED) return '';
  const at = DRAFT_CHECKED.toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit' });
  return `<span class="checkedAt muted small">checked at ${esc(at)}</span>`;
}

// Whether the verdict and its buttons are the bar at the foot of the screen or the end
// of the panel they are written in. Read off the bar, so the width it changes at is
// stated in the stylesheet and nowhere else.
function footbarFixed() {
  const bar = document.getElementById('dActions');
  return !!bar && typeof window.getComputedStyle === 'function'
    && window.getComputedStyle(bar).position === 'fixed';
}

// A fixed bar takes no space in the flow, so the end of the editor would sit under it.
// Give the page exactly the bar's height back.
function reserveFootbar() {
  const grid = document.querySelector('.designer');
  if (!grid) return;
  const bar = document.getElementById('dActions');
  grid.style.paddingBottom = footbarFixed() ? (bar.offsetHeight + 14) + 'px' : '';
}

// What the save said, brought to where the press was. The button is in the bar and the
// answer lands in the panel behind it, which on a phone is several screens away.
function showSaved(box) {
  if (footbarFixed() && box.scrollIntoView) box.scrollIntoView({ block: 'center' });
}

// Turning a phone on its side changes both the bar's height and whether there is one.
function wireFootbar() {
  if (wireFootbar.done) return;
  wireFootbar.done = true;
  window.addEventListener('resize', () => reserveFootbar());
}

// ------------------------------------------------------ from a draft to a number

// Where the draft is on the rungs, and its family. Asked once per hash, because a name
// or a description is not a new game.
function loadStanding(hash) {
  const query = '?parent=' + encodeURIComponent(DRAFT_FROM || '')
    + '&name=' + encodeURIComponent(DRAFT.name || '');
  return api('/api/design/standing/' + hash + query).then(standing => {
    if (hash !== DRAFT_HASH) return;
    const before = DRAFT_STANDING && DRAFT_STANDING.hash === hash ? DRAFT_STANDING : null;
    const was = !!(before && before.measuring && before.measuring.state === 'running');
    DRAFT_STANDING = standing;
    redrawLadder();
    // A score measured on this computer and held until the game was saved goes now (Q1004).
    hereStood(standing);
    const run = standing.measuring;
    const running = !!(run && run.state === 'running');
    if (running) pollStanding(hash);
    // A measurement just ended on the game the draft came from: what that game is was
    // fetched before it had a number, so it is asked again for the rows that read it.
    if (was && !running && DRAFT_PARENT && DRAFT_PARENT.hash === hash) {
      api('/api/design/' + hash).then(source => {
        if (DRAFT_PARENT && DRAFT_PARENT.hash === source.hash) {
          DRAFT_PARENT = source;
          redrawLadder();
        }
      }).catch(() => {});
    }
  }).catch(() => {});
}

// The screen is cheap enough to run on every new hash, like the smoke gate.
function runScreen(hash) {
  api('/api/design/screen', { spec: DRAFT }).then(screen => {
    if (hash !== DRAFT_HASH) return;
    DRAFT_SCREEN = screen;
    redrawLadder();
  }).catch(error => {
    if (hash !== DRAFT_HASH) return;
    DRAFT_SCREEN = { hash, passed: false, reasons: [errorWords(error)] };
    redrawLadder();
  });
}

// A measurement runs for minutes on a thread; the rungs follow it until it ends.
function pollStanding(hash) {
  if (STANDING_TIMER) return;
  STANDING_TIMER = setInterval(() => {
    const run = (DRAFT_STANDING || {}).measuring;
    if (hash !== DRAFT_HASH || !document.getElementById('dLadder')
        || !run || run.state !== 'running') {
      clearInterval(STANDING_TIMER); STANDING_TIMER = null;
      if (hash === DRAFT_HASH) loadStanding(hash);
      return;
    }
    loadStanding(hash);
  }, 3000);
}

function redrawLadder() {
  const ladder = document.getElementById('dLadder');
  if (ladder && DRAFT_CHECK && DRAFT_CHECK.valid) ladder.innerHTML = ladderHtml(DRAFT_CHECK);
  const family = document.getElementById('dFamily');
  if (family) family.innerHTML = familyHtml();
  // *Which game this is* reads the standing too, once the draft is the saved game.
  const identity = document.getElementById('dIdentity');
  if (identity && DRAFT_CHECK && DRAFT_CHECK.valid) {
    identity.outerHTML = identityHtml(DRAFT_CHECK);
  }
  // The measuring control moves into *Publish* once the standing says the button is this
  // person's to press, and back out when it is not (`hereInPanel`).
  const away = document.getElementById('dHereAway');
  if (away && DRAFT_CHECK && DRAFT_CHECK.valid) away.innerHTML = hereAwayHtml(DRAFT_CHECK);
  wireLadder();
  wireHere();
  waitForPublish();
}

// The standing the rungs are drawn from, when it is the one for this check's hash.
function standingFor(check) {
  return check && DRAFT_STANDING && DRAFT_STANDING.hash === check.hash ? DRAFT_STANDING : null;
}

// Q742, the user: the measuring control sits where the measurement is asked for, so a
// designer never scrolls away to measure and back to publish. On the site, wherever the
// standing draws this person's *Publish* button, the control -- its button, its progress
// and its result -- is drawn inside that panel, straight under the line that says a score
// is needed (`publishHtml`). Everywhere else (the local application, somebody else's
// game, a game already on trial) it keeps its own place under the rungs.
function hereInPanel(s) {
  return ACCOUNTS && !!(s && s.publish && s.publish.publishable);
}

function hereAwayHtml(check) {
  return hereInPanel(standingFor(check)) ? '' : `<div id="dHere">${hereHtml(check)}</div>`;
}

// While a press of *Publish* waits for the site's check (Q742), the standing is asked
// again every few seconds, so the panel turns to *On trial* by itself when the runner
// puts it there, with nothing more to press. One timer, gone once nothing waits.
let PUBLISH_TIMER = null;
function waitForPublish() {
  const p = (DRAFT_STANDING || {}).publish;
  const waiting = !!(p && p.waitsForCheck) && DRAFT_STANDING.hash === DRAFT_HASH;
  if (!waiting || PUBLISH_TIMER) return;
  const hash = DRAFT_HASH;
  PUBLISH_TIMER = setTimeout(() => {
    PUBLISH_TIMER = null;
    if (hash === DRAFT_HASH && document.getElementById('dPublish')) loadStanding(hash);
  }, PUBLISH_POLL);
}
let PUBLISH_POLL = 5000;

// Each rung is done, failed, under way or still ahead, with what it costs on this
// machine. A number appears only on the measurement rung, and only when it is current.
function rungStates(check, standing, screen) {
  const s = standing || {};
  const saved = !!s.saved;
  const m = s.measured || { state: 'unscored' };
  const run = s.measuring && s.measuring.state === 'running' ? s.measuring : null;
  const states = {};
  states.validate = { mark: 'done', says: 'yes' };
  states.smoke = check.playable === false
    ? { mark: 'failed', says: (check.playableReasons || []).join('; ') }
    : check.playable === null
      ? { mark: 'failed', says: (check.playableReasons || []).join('; ') }
      : { mark: 'done', says: `yes (${Number(check.smokeSeconds || 0).toFixed(2)} seconds)` };
  states.screen = !screen || screen.hash !== check.hash
    ? { mark: 'running', says: 'running...' }
    : screen.passed
      ? { mark: 'done', says: `yes (${Number(screen.seconds || 0).toFixed(2)} seconds)` }
      : { mark: 'failed', says: (screen.reasons || []).join('; ') };
  states.save = !standing
    ? { mark: 'running', says: 'asking...' }
    : saved
      ? { mark: 'done', says: savedWords(s) }
      : { mark: 'ahead', says: ACCOUNTS && !VIEWER ? 'sign in to save it' : 'not yet' };
  if (run) {
    states.canonical = { mark: 'running', says: measuringWords(run) };
  } else if (m.state === 'current') {
    states.canonical = ACCOUNTS
      ? { mark: 'done', says: `scored ${measuredNumberHtml(m, ', the ')} (${
        esc(rulesVersion(m.weights || '?'))})${m.viable === false ? '; ruled out' : ''}`,
      html: true }
      : { mark: 'done', says: `measured, ${measuredNumberHtml(m)},
      under weights ${esc(m.weights || '?')}${m.viable === false ? ', not viable' : ''}`, html: true };
  } else if (m.state === 'stale') {
    states.canonical = { mark: 'ahead', says: ACCOUNTS
      ? 'measured, but out of date: no score is shown until it is measured again'
      : `measured, but out of date; needs a ${m.needs}` };
  } else {
    states.canonical = { mark: 'ahead', says: saved ? 'not yet' : 'save it first' };
  }
  states.nominate = s.stage === 'committed'
    ? { mark: 'done', says: ACCOUNTS ? 'already one of them' : 'already part of the yardstick' }
    : s.nominated
      ? { mark: 'done', says: ACCOUNTS
        ? 'put forward; the person who runs the site decides'
        : 'nominated; a maintainer decides' }
      : { mark: 'ahead', says: saved ? 'not yet' : 'save it first' };
  return states;
}

// What the save rung says once the draft is saved (D20). A saved draft is *your draft*;
// rules already in the library say so by name. The local application keeps the stage.
function savedWords(s) {
  if (!ACCOUNTS) return s.stage === 'candidate' ? 'yes, as a candidate' : 'yes, ' + s.stage;
  if (s.stage === 'committed' || s.stage === 'archived') {
    return 'yes: these rules are already a library game' + (s.name ? ', ' + s.name : '');
  }
  return 'yes, as your draft';
}

// A running measurement in words. Queued, starting, or under way with the last thing it
// said; the time is said only once it is its own, because a queued run's clock is the
// wait and a starting run's is nought. `run` is `keeping.Measurement.as_dict`.
function measuringWords(run) {
  if (run.waiting) return 'queued behind another measurement; it starts when that one ends';
  const last = String((run.log || []).slice(-1)[0] || '').trim();
  const seconds = Math.round(run.seconds || 0);
  if (!last && seconds < 1) return 'starting';
  const so = `measuring for ${seconds} second${seconds === 1 ? '' : 's'} so far`;
  return last ? `${so}; last step: ${last}` : so;
}

const RUNG_MARKS = { done:'&#10003;', failed: '&#10007;', running: '&hellip;', ahead: '&#9675;' };

// A rung in this server's words. The local application's are the maintainer's -- the
// command line, `runs/designed/`, the yardstick -- and each rung carries a visitor's
// beside them (`candidacy.SITE_WORDS`), drawn where this is the site (Q576, F6).
function rungWords(rung) {
  return ACCOUNTS && rung.site ? Object.assign({}, rung, rung.site) : rung;
}

function ladderHtml(check) {
  if (!check || !check.valid) return '';
  const standing = DRAFT_STANDING && DRAFT_STANDING.hash === check.hash ? DRAFT_STANDING : null;
  const rungs = ((standing && standing.rungs) || (DESIGN_KIT || {}).rungs || []).map(rungWords);
  const states = rungStates(check, standing, DRAFT_SCREEN);
  const s = standing || {};
  const run = s.measuring && s.measuring.state === 'running';
  const current = (s.measured || {}).state === 'current';
  const measurable = s.saved && !run && !current && !s.build;
  const nominable = s.saved && !s.nominated && s.stage !== 'committed';
  const canMeasure = measurable && offered('POST', '/api/design/measure');
  const canNominate = nominable && offered('POST', '/api/design/nominate');
  // Said only about measuring (D23). Nominating for the Baseline is the local
  // application's step alone, and the site's shortlist has a *nominate it* of its own that
  // works, so a sentence saying the site nominates nothing would contradict a live button
  // drawn just above it on a game on trial.
  const withheld = measurable && !canMeasure;
  return `${publishHtml(s, check)}<h3 class="u-mt-0">From a draft to a ${
      ACCOUNTS ? 'score' : 'number'}</h3>
    <ol class="rungs small u-m-0 u-pl-0 u-ls-none">${rungs.map(rung => {
      const state = states[rung.key] || { mark: 'ahead', says: '' };
      return `<li data-rung="${esc(rung.key)}" data-state="${state.mark}" class="u-mb-7">
        <span class="u-d-inline-block u-w-1_4em ${state.mark === 'done'
          ? 'u-c-good' : state.mark === 'failed' ? 'u-c-bad' : 'u-c-muted'}"
          >${RUNG_MARKS[state.mark]}</span><b>${esc(rung.name)}</b> &mdash;
        ${state.html ? state.says : esc(state.says)}
        <div class="muted u-ml-1_4em">${esc(rung.what)}. ${ACCOUNTS
          ? `Takes ${esc(rung.cost)}; ${esc(rung.runs)}.`
          : `Costs ${esc(rung.cost)}; runs ${esc(rung.runs)}.`}</div></li>`;
    }).join('')}</ol>
    ${s.build ? `<div class="muted small">${ACCOUNTS
      ? 'Measuring is paused while the site is updated.'
      : 'Measuring is held: ' + esc(s.build)}</div>` : ''}
    <div class="row u-mt-8">
      ${canMeasure ? `<button id="dMeasure" class="primary"
        title="queued behind any measurement already running">measure it</button>` : ''}
      ${canNominate ? `<button id="dNominate">nominate it for the yardstick</button>` : ''}
    </div>
    ${canNominate ? `<textarea id="dNominateNote" class="u-mt-6 u-minh-44"
      placeholder="What does this game show that the others do not? A maintainer reads this."></textarea>` : ''}
    ${withheld ? `<div class="muted small u-mt-6">The site does not measure
      designed games itself yet; a measurement takes minutes of computer time.</div>` : ''}
    ${promotionHtml(s)}
    <div class="muted small u-mt-6">${ACCOUNTS
      ? 'The editor never changes the games every score is measured against.'
      : `The editor never writes to the yardstick${info('committed')}. A nomination asks a
         maintainer, who promotes with a reason.`}</div>
    <div id="dLadderSays"></div>
    ${nominationsHtml()}`;
}

// *Where this game stands* (§3.1): the level and the four steps with it lit, the one word
// every page uses (`site/stages.py`, Q503b), then Q503a's *Publish*. On the site a save
// keeps a draft and *Publish* lists it: the server sends where the draft stands
// (`drafts.standing`), the two ticks a publish needs, whether the button works and, while
// it cannot, why. Only its designer sees the ticks and the button. A game the store holds
// that nobody designed here (a Library or Baseline game) has the track and no *Publish*.
// The local application sends neither, and this draws nothing. On trial, its designer
// also sees the shortlist's bar and *nominate it* (Q518), drawn by the game page's own
// `shortlistHtml` from the same `shortlist.panel` the game page is sent.
//
// Q742: the button opens once this computer has measured the draft and sent the score,
// not when the site has finished checking it. A press then waits (`waitsForCheck`): the
// button gives way to *Waiting for verification* and the server's sentence, and the panel
// is asked again until the game is on trial (`waitForPublish`). A waiting publish the site
// called off says why (`stopped`) above the button. The measuring control is drawn here,
// under the line that asks for a score (`hereInPanel`).
function publishHtml(s, check) {
  const p = s.publish;
  const level = s.level || (p && p.level) || '';
  if (!p && !s.track) return '';
  const waits = !!(p && p.waitsForCheck);
  // A list of `.marks`: the mark in a column of its own, so a tick's sentence that wraps
  // stays clear of it on a phone (Q777).
  const ticks = ((p && p.ticks) || []).map(tick => `<li class="u-mb-4"><span
      class="${tick.met ? 'u-c-good' : 'u-c-bad'}">${tick.met ? '&#10003;' : '&#10007;'}</span>${
      esc(tick.says)}</li>`).join('');
  // Q722: the terms' one sentence beside a box, the box unticked each time it is drawn.
  // The button waits for it as well as for the ticks; a press sends the version the box
  // stood for and nothing else (`site/terms.py`).
  const terms = p && p.publishable && p.terms && !waits ? p.terms : null;
  const agree = terms ? `<label class="small u-d-flex u-gap-8 u-ai-flex-start u-m-6-0-8 u-maxw-34rem" id="dTerms">
        <input type="checkbox" id="dTermsAgree" data-version="${esc(String(terms.version))}" class="u-mt-3 u-flex-none">
        <span>${esc(terms.says)} <a href="${esc(terms.href)}" target="_blank" rel="noopener"
          >the terms</a></span></label>` : '';
  const button = p && p.publishable && !waits
    ? `${agree}<div class="row"><button id="dPublishGo" class="primary" disabled
        data-open="${p.open ? 'yes' : 'no'}"
        title="${esc(p.open ? 'tick the box to agree to the terms first'
          : p.waiting || 'both ticks are needed first')}"
        >publish</button></div>` : '';
  const says = p ? p.says : s.levelSays;
  const waiting = !p || !p.waiting ? ''
    : waits ? `<div class="verdict u-m-8-0" id="dPublishWaiting">
        <div>Waiting for verification</div>
        <div class="small u-fw-400 u-mt-5">${esc(p.waiting)}</div></div>`
    : `<div class="small u-m-4-0-6" id="dPublishWaiting">${esc(p.waiting)}</div>`;
  // While a press waits there is nothing to measure: the control is drawn only to show a
  // measurement already in hand (its progress, its number, the site's check of it).
  const here = hereInPanel(s) && !(waits && !HERE) ? `<div id="dHere" data-in="panel">${
    hereHtml(check === undefined ? DRAFT_CHECK : check, true)}</div>` : '';
  return `<div id="dPublish" data-level="${esc(level)}" class="u-m-0-0-12">
      <h3 class="u-mt-0">Where this game stands: ${esc(level)}</h3>
      ${trackHtml(s.track)}
      ${says ? `<div class="small">${esc(says)}</div>` : ''}
      ${ticks ? `<ul class="small marks u-m-6-0">${ticks}</ul>` : ''}
      ${p && p.stopped ? `<div class="error small u-m-6-0" id="dPublishStopped">${
        esc(p.stopped)}</div>` : ''}
      ${waiting}
      ${here}
      ${button}
      ${s.shortlist && s.shortlist.mine ? shortlistHtml(s.shortlist) : ''}
      <div id="dPublishSays"></div>
    </div>`;
}

// What promotion needs, in the page's words (`candidacy.promotion`): how many runs are
// measured, what they give the game here, and that promotion wants four. The command a
// maintainer runs for the four is folded away, and only where this copy measures at
// all: on the site nobody reading it has a terminal to type it into. On the site none of
// it is drawn (Q576, F6): promotion there is the track's (*Where this game stands*), and
// a maintainer's four runs of 256 games are nothing a visitor can act on.
function promotionHtml(s) {
  const p = s.promotion;
  if (ACCOUNTS || !p || !p.says || s.stage === 'committed') return '';
  const local = offered('POST', '/api/design/measure');
  return `<div class="muted small u-mt-6" id="dPromotion">
      ${esc(p.says)}${local && p.command
        ? `<details class="u-mt-4"><summary>for whoever keeps this copy</summary>
            From a terminal in the project, this measures it for promotion:
            <code>${esc(p.command)}</code></details>` : ''}</div>`;
}

function wireLadder() {
  const say = html => {
    const box = document.getElementById('dLadderSays');
    if (box) box.innerHTML = html;
  };
  const hash = DRAFT_HASH;
  const measure = document.getElementById('dMeasure');
  if (measure) measure.onclick = () => {
    measure.disabled = true;
    api('/api/design/measure', { hash }).then(run => {
      if (DRAFT_STANDING && DRAFT_STANDING.hash === hash) DRAFT_STANDING.measuring = run;
      redrawLadder();
      pollStanding(hash);
    }).catch(error => {
      measure.disabled = false;
      say(`<div class="error">${esc(errorWords(error))}</div>`);
    });
  };
  const publish = document.getElementById('dPublishGo');
  const agree = document.getElementById('dTermsAgree');
  if (publish && agree) agree.onchange = () => {
    const open = publish.dataset.open === 'yes';
    publish.disabled = !(open && agree.checked);
    if (open) publish.title = agree.checked ? 'put it on trial'
      : 'tick the box to agree to the terms first';
  };
  if (publish) publish.onclick = () => {
    publish.disabled = true;
    const box = document.getElementById('dPublishSays');
    const body = { hash };
    if (agree && agree.checked) body.terms = Number(agree.dataset.version);
    api('/api/design/publish', body).then(answer => {
      if (box) box.innerHTML = `<div class="ok">${esc(answer.said)}</div>`;
      // Q742: a press the site takes while it checks the score comes back `waiting`, and
      // the panel redrawn from the standing says so in its own box, in the same words.
      loadStanding(hash).then(() => {
        const again = document.getElementById('dPublishSays');
        if (again && !answer.waiting) again.innerHTML = `<div class="ok">${esc(answer.said)}</div>`;
      });
    }).catch(error => {
      if (box) box.innerHTML = `<div class="error">${esc(errorWords(error))}</div>`;
    });
  };
  // Q518. The track's *nominate it*: the game page's route, then the standing asked again,
  // so the track and its sentence are the server's new words.
  const shortlisting = document.querySelector('#dPublish #nominateIt');
  if (shortlisting) shortlisting.onclick = () => {
    shortlisting.disabled = true;
    const box = document.getElementById('dPublishSays');
    api('/api/game/' + hash + '/nominate', {}).then(answer => {
      loadStanding(hash).then(() => {
        const again = document.getElementById('dPublishSays');
        if (again) again.innerHTML = `<div class="ok">${esc(answer.said)}</div>`;
      });
    }).catch(error => {
      shortlisting.disabled = false;
      if (box) box.innerHTML = `<div class="error">${esc(errorWords(error))}</div>`;
    });
  };
  const nominate = document.getElementById('dNominate');
  if (nominate) nominate.onclick = () => {
    const note = (document.getElementById('dNominateNote') || {}).value || '';
    api('/api/design/nominate', { hash, note }).then(answer => {
      if (DESIGN_KIT) DESIGN_KIT.nominations = answer.nominations;
      loadStanding(hash);
    }).catch(error => say(`<div class="error">${esc(errorWords(error))}</div>`));
  };
}

// What maintainers are asked to look at. Each carries its blocker and the command that
// promotes it, because promotion is theirs and happens outside the editor. A checkout's
// list, so the site draws none: its nominations are the shortlist's.
function nominationsHtml() {
  const listed = (DESIGN_KIT || {}).nominations || [];
  if (ACCOUNTS || !listed.length) return '';
  return `<details class="u-mt-10"><summary class="muted small">nominated for the
      yardstick (${listed.length})</summary>
    <ul class="rules small u-mt-6">${listed.map(n => `<li>
      <b>${esc(n.name)}</b> <code>${esc(n.hash)}</code> &mdash; ${esc(n.stage)}${
        n.note ? `<div class="muted">${esc(n.note)}</div>` : ''}${
        n.blocker ? `<div class="muted">${esc(n.blocker)}</div>`
          : n.command ? `<div><code>${esc(n.command)}</code></div>` : ''}</li>`).join('')}
    </ul></details>`;
}

// The family as a tree from its oldest known ancestor. Every save is a node with its own
// hash and its own measurement state, so four saves of one idea are four games.
function familyHtml() {
  const standing = DRAFT_STANDING;
  if (!standing || !standing.lineage || standing.hash !== DRAFT_HASH) return '';
  const tree = standing.lineage.root;
  return `<h3 class="u-mt-0">Its family</h3>
    ${lineageHtml(tree)}
    ${standing.lineage.cut ? '<div class="muted small">The family is larger than is drawn here.</div>' : ''}
    <div class="muted small u-mt-6">${ACCOUNTS
      ? `Each save that changes a rule is a game of its own. A score belongs to the exact
         rules it was measured on; an out-of-date score is not shown.`
      : `Each save is its own game with its own hash${info('rulesetHash')}. A number
         belongs to the hash it was measured on, and a stale one is not shown.`}</div>`;
}

// The hash beside each game is the maintainer's (Q576, F6): a visitor reads the name.
function lineageHtml(node) {
  if (!node) return '';
  const label = node.draft
    ? `<i>${esc(node.name || 'this draft')}</i>`
    : node.known
      ? `<a href="/game/${esc(node.hash)}">${esc(node.name || (ACCOUNTS ? 'a game with no name' : node.hash))}</a>`
      : `<span class="muted">a game this library cannot find</span>`;
  const shown = `${node.here ? '<b>' : ''}${label}${node.here ? '</b>' : ''}`;
  return `<ul class="lineage small u-m-4-0-0 u-pl-16">
    <li data-lineage="${esc(node.hash)}"${node.here ? ' data-here="1"' : ''}>${shown}
      ${ACCOUNTS ? '' : `<code>${esc(node.hash)}</code> `}&mdash; ${lineageStateHtml(node)}${
      (node.children || []).map(lineageHtml).join('')}</li></ul>`;
}

// A current measurement's number and the runs it rests on, in the words a card's hover
// uses, for the measurement rung and for each game in the lineage. `m` is
// `candidacy.measured`. A current measurement can still have no number -- too few of its
// runs could be scored -- and that reads *no score*, never the 0.00 `Number(null)` makes;
// a number with no count is not drawn either, because it would not say which claim it is.
// `joint` is what stands between the number and its runs: *0.62 as the mean of four
// runs*, or on the site's ladder *0.62, the average of four runs* (D21).
function measuredNumberHtml(m, joint = ' as the ') {
  const rested = runsPhrase(m.seeds);
  if (m.index === null || m.index === undefined || !rested) {
    return '<span class="muted">no score</span>';
  }
  return `<b>${Number(m.index).toFixed(2)}</b>${joint}${esc(rested)}`;
}

function lineageStateHtml(node) {
  if (node.draft) return '<span class="muted">this draft, not saved</span>';
  const m = node.measured || { state: 'unscored' };
  const nominated = node.nominated ? ' <span class="muted">(nominated)</span>' : '';
  if (m.state === 'current') {
    return `measured, ${measuredNumberHtml(m)}${nominated}`;
  }
  if (m.state === 'stale') {
    return `<span class="muted">measured, but out of date${ACCOUNTS
      ? '' : '; needs a ' + esc(m.needs)}</span>${nominated}`;
  }
  return `<span class="muted">not measured</span>${nominated}`;
}

// What this draft *is*, which is a hash over its mechanics and nothing else -- and what
// that costs, which is the whole reason the editor has to say it. A measurement is filed
// under the hash it was taken on, so changing one rule does not make a changed game with
// an old score: it makes a different game with no score at all. Nobody discovers that
// from a save button.
// What a fork changed from the game it started as, in the same sentences the rules
// below are written in. The server compares them (`describe_changes`); the page only
// lists what it was told, and says nothing for a draft with no parent to compare with.
function changesHtml(check) {
  const changes = check.changes;
  if (!DRAFT_FROM || !Array.isArray(changes)) return '';
  const parent = (DRAFT_PARENT && DRAFT_PARENT.name) || nameOf(DRAFT_FROM) || DRAFT_FROM;
  return `<div class="panel u-mt-14" id="dChanges">
    <h3 class="u-mt-0">What you changed from ${esc(parent)}</h3>
    ${changes.length
      ? `<ul class="rules small">${changes.map(c => `<li>${esc(c)}</li>`).join('')}</ul>`
      : `<div class="muted small">Nothing yet: every rule still reads as ${esc(parent)}'s does.</div>`}
  </div>`;
}

function identityHtml(check) {
  const parent = DRAFT_PARENT;
  const same = parent && parent.hash === check.hash;
  const rows = [];
  if (same) {
    // One game, so one row: a hash listed twice read as two games, one of them never
    // measured. Just after a save the parent *is* the save, under the same name.
    const named = DRAFT.name || parent.name;
    const also = parent.name && parent.name !== named
      ? `<b>the same game</b> as ${esc(parent.name)}; ` : '';
    rows.push([esc(named), check.hash, also + identityMeasuredHtml(check, parent)]);
  } else {
    if (parent) rows.push([esc(parent.name), parent.hash, measuredHtml(parent)]);
    rows.push([esc(DRAFT.name || 'this draft'), check.hash,
      `<span class="muted">no ${ACCOUNTS ? 'score' : 'measurement'} of its own</span>`]);
  }
  // The hash is the maintainer's word for which game this is (Q576, F6). On the site the
  // rows are the names alone, and the sentence says what the hash stands for instead.
  const listed = ACCOUNTS
    ? `<div>${rows.map(([name, , note]) =>
        `<div class="small u-mb-3">${name} &mdash; ${note}</div>`).join('')}</div>`
    : `<div class="identity">${rows.map(([name, hash, note]) =>
        `<code>${esc(hash)}</code><span>${name} &mdash; ${note}</span>`).join('')}</div>`;
  return `<div class="panel u-mt-14" id="dIdentity">
    <h3 class="u-mt-0">Which game this is</h3>
    ${listed}
    <div class="muted small u-mt-9">
      ${ACCOUNTS ? 'A game is its rules' : `A game is identified by a hash over its
      mechanics${info('rulesetHash')}`} &mdash; the
      board, the pieces, how they move, where they start and how it ends. A name, a
      description or a drawing is none of those, so changing one changes nothing above.
      ${same
        ? `Nothing you have changed is a rule yet, so this <b>is</b> ${esc(parent.name)}
           and saving it would not make a second game.`
        : ACCOUNTS
          ? `${parent ? `Changing a rule made a different game. ` : ''}This game has no
             score yet${parent ? `: a score from ${esc(parent.name)}'s rules says nothing
             about these` : ''}. No score is not a score of zero.`
          : parent
          ? `Changing a rule made a different game. ${esc(parent.name)} keeps its own
             measurement; this one has none until somebody measures it, and a number
             taken on ${esc(parent.hash)} is not a number about this.`
          : `Nothing has measured it, so it has no number rather than a zero.`}
    </div>
  </div>`;
}

// The measurement of the game a draft still is. The standing is asked again while a
// measurement runs and when it ends, so it is the fresh answer; the parent was fetched
// once, at the save or the fork, and said *never measured* after the rung said measured.
function identityMeasuredHtml(check, parent) {
  const s = DRAFT_STANDING;
  if (s && s.hash === check.hash && s.saved) {
    if (s.measuring && s.measuring.state === 'running') {
      return '<span class="muted">being measured now</span>';
    }
    return lineageStateHtml({ measured: s.measured });
  }
  return measuredHtml(parent);
}

// Whether the game this draft came from has a number, in the interface's own three
// states: current, stale with the repair named, or never measured. Elsewhere a stale
// score is shown as its last number, marked provisional (EV2, docs/archive/plan-evolving-score.md
// §3). Here it reads *out of date* with no number, by design: the editor compares a draft
// with its parent, a provisional estimate enters no comparison, and the server sends the
// parent's `current_score` only (`/api/design/<hash>`). A current measurement
// with no number -- too few of its runs could be scored -- or with no run count reads
// *no score*, not *never measured*: it was measured, and a number that does not say
// what it rests on is not drawn.
function measuredHtml(parent) {
  const rested = runsPhrase(parent.seeds);
  if (parent.index !== null && parent.index !== undefined && rested) {
    return ACCOUNTS
      ? `score <b>${parent.index.toFixed(2)}</b>, <span class="muted">the ${
        esc(rested)}</span>`
      : `measured, <b>${parent.index.toFixed(2)}</b>
      <span class="muted">on ${(parent.roster || []).length} players, ${
        esc(rested)}</span>`;
  }
  const stale = parent.staleness;
  if (stale && !stale.current) {
    return `<span class="muted">measured, but out of date${ACCOUNTS ? '' : ` &mdash; needs a ${
      esc(stale.needs || 'replay')}`}</span>`;
  }
  if (stale) return 'measured, <span class="muted">no score</span>';
  return '<span class="muted">never measured</span>';
}

// What the marks on the board mean, and which of the two questions the engine was
// asked. On an empty board they are what the descriptors say; on the opening position
// they are what this piece may actually do from where it stands.
function moveKeyHtml() {
  const map = DRAFT_MAP;
  if (!map) return '';
  const piece = (DRAFT.pieces || [])[DRAFT_PIECE] || {};
  const called = esc(piece.name || piece.id || 'this piece');
  const dots = `<span class="u-c-dot">&bull;</span> to move and
    <span class="u-c-bad">&#9711;</span> to take`;
  const W = map.width || 1;
  const here = map.from >= 0 ? squareName(map.from % W, Math.floor(map.from / W)) : '';
  // A wild piece's type (the wolves walk, step 12): the wild one is on the board but is
  // nobody's, so the answer is where its walk is written, not that none stands there.
  const wild = map.live ? wildOnesWords(piece.id) : [];
  const wildSaid = wild.length ? `The wild <b>${called}</b>: ${wild.join('; ')}. A wild
    piece moves by itself after every move, and its walk is written on the
    ${wildCardLinkHtml()} card.` : '';
  if (map.live && map.from < 0) {
    if (wild.length) {
      return `<div class="muted small u-mt-6" data-wildgoes>${wildSaid}
        No player's <b>${called}</b> stands on the board in the opening position; one put
        down under <i>set up</i> would be a player's.</div>`;
    }
    return `<div class="muted small u-mt-6">No <b>${called}</b> stands on
      the board in the opening position, so there is nothing for it to do yet. Put one
      down under <i>set up</i>.</div>`;
  }
  if (map.live) {
    return `<div class="muted small u-mt-6">
      The marks show every move the <b>${called}</b> on ${here} may make in the opening
      position, with ${dots}. Its own army, the pieces in its way and every other rule of
      this game are in that answer${map.standing.length > 1
        ? `. Click another ${called} to ask about that one` : ''}.${ACCOUNTS
        ? '' : ' The engine answered it, not the page.'}${wildSaid
        ? `<div data-wildgoes class="u-mt-4">${wildSaid}</div>` : ''}</div>`;
  }
  return `<div class="muted small u-mt-6">
    The marks show where <b>${called}</b> could go from ${here} on an empty board, with
    ${dots}${map.captureOnly.length
      ? ', and it may only take on the ringed ones' : ''}.${ACCOUNTS
      ? '' : ' The engine answered this, not the page.'}</div>`;
}

// The rungs a save's check runs, in order, as the server keys them.
const SAVE_RUNGS = ['validate', 'smoke', 'screen'];

// A rung's name in the ladder's own words, so a sentence about a rung and the ladder
// below it cannot drift apart. An unknown key reads as itself rather than as nothing.
function rungName(key) {
  const rungs = (DESIGN_KIT || {}).rungs || [];
  const found = rungs.find(rung => rung.key === key);
  return found ? rungWords(found).name : key;
}

// How far the check the save ran got, and whether the draft passed there. Only the site
// runs that check, so a local save says nothing and this returns the empty string.
function saveRungHtml(kept) {
  if (!kept.rung) return '';
  if (kept.rungPassed) {
    // D26: the check's three steps, by count; each is named on the ladder below.
    return `The playability check ran all ${countWord(SAVE_RUNGS.length)} of its steps,
      and these rules passed every one.`;
  }
  return `The playability check got as far as <b>${esc(rungName(kept.rung))}</b>, and
    these rules did not pass there.`;
}

// The save's first sentence: kept, kept already, or kept already with new words (Q595).
// The rules are the game's hash and the name and description are not, so saving kept
// rules with a new description is the same game with the new description -- said as
// that, not as *Already saved*, which read as though the description had been dropped.
// The site keeps a published game's words, or another member's, and says why (`keptAsIs`).
function savedWordsHtml(kept) {
  const name = `<b>${esc(kept.name)}</b>`;
  if (kept.created) return `Saved as ${name}.`;
  const restated = kept.restated || [];
  if (restated.length) {
    const what = restated.includes('name')
      ? (restated.includes('description') ? 'name and description' : 'name')
      : 'description';
    return `Saved the new ${what} of ${name}. Its rules are the same, so it is the
      same game.`;
  }
  const asIs = (kept.unchanged || []).length && kept.keptAsIs ? ` ${esc(kept.keptAsIs)}` : '';
  return `Already saved as ${name}.${asIs}`;
}

// What the save did, beside the button that did it: whether the draft was kept, how far
// the check got, and whether it was listed and why. The site answers the last two and
// the local application, which has no listing, answers neither -- so each sentence is
// drawn only where the save's answer carries it.
function savedHtml(kept) {
  const rung = saveRungHtml(kept);
  return `<div class="ok u-mt-10">
    ${savedWordsHtml(kept)} A change to its rules from here is a new game, and its
    family will show it.
    <a href="/game/${esc(kept.hash)}">open its page</a> ${copyLinkHtml(kept.hash)}
    ${rung ? `<div class="u-mt-6 ${kept.rungPassed
      ? 'u-c-ink' : 'u-c-bad'}">${rung}</div>` : ''}
    ${kept.said ? `<div class="u-mt-6 u-c-ink">${esc(kept.said)}</div>` : ''}
  </div>`;
}

// A hint about the saved description, optional and never waited for (Q396). Asked only
// once the save has answered and been drawn. The server reads the saved description a
// sentence at a time against the rules, through TypeSafe's Jev, when it has a key. With
// none it says it is off and this page stops asking; with nothing to say, a failure, or
// the day's requests spent, nothing is drawn. The hint joins the save's own message, so a
// redraw keeps it and an edit to the rules clears both.
let HINT_OFF = false;

function askHint(hash) {
  if (HINT_OFF) return;
  const saved = DRAFT_SAVED;
  api('/api/design/hint', { hash }).then(answer => {
    if (answer.off) { HINT_OFF = true; return; }
    const claims = answer.claims || [];
    if (!claims.length || DRAFT_SAVED !== saved) return;
    DRAFT_SAVED = saved + hintHtml(claims);
    const box = document.getElementById('dSaved');
    if (box) box.innerHTML = DRAFT_SAVED;
  }).catch(() => {});
}

// The sentences as the description has them, and what the reading is: a model's, which
// can be wrong, and which changed nothing.
function hintHtml(claims) {
  return `<div class="muted small u-mt-8" data-hint="description">
    <b>A hint about the description.</b> ${claims.length === 1
      ? 'This sentence in its saved description seems' : 'These sentences in its saved description seem'}
    not to match the rules:
    <ul class="rules u-m-4-0">${claims.map(one =>
      `<li>&ldquo;${esc(one.claim)}&rdquo;</li>`).join('')}</ul>
    An automatic check by an AI model (TypeSafe's Jev), which can be wrong. It changed nothing:
    the game is saved either way.</div>`;
}

// ------------------------------------------------ a save under a borrowed name
//
// Q995c (fix 9, F7). A draft built from a template keeps the template's name, and was saved
// as *skirmish* with nothing said: the library then held a second *skirmish* that was not
// the template. A save under a template's name, or under the name of another game the
// library lists, asks for a name of its own first, once; keeping the name is one press.

let NAME_KEPT = '';   // a name kept on purpose or already saved under, in lower case

// Whose name the draft is still called by: 'template', 'game', or '' when it is its own.
function borrowedName() {
  const name = String(DRAFT.name || '').trim().toLowerCase();
  if (!name || name === NAME_KEPT) return '';
  const same = other => String(other || '').trim().toLowerCase() === name;
  if (((DESIGN_KIT && DESIGN_KIT.templates) || []).some(t => same((t.spec || {}).name))) {
    return 'template';
  }
  // The game a copy came from is the commonest clash of all: *chess*, saved as *chess*.
  // A name the draft has been saved under is its own (`NAME_KEPT`, set by the save).
  if (LIBRARY.some(game => game && same(game.name))) return 'game';
  return '';
}

function nameAskHtml(whose) {
  const name = String(DRAFT.name || '').trim();
  return `<div class="nameAsk small" role="group" aria-label="a name of its own">
    <div><b>This draft is still called '${esc(name)}'</b>, ${whose === 'template'
      ? 'the name of the template it started from'
      : 'the name of a game already in the library'}. A saved game is listed by its
      name, so give it one of its own first.</div>
    <label class="field u-m-8-0-0"><span>what it is called</span>
      <input id="dSaveName" placeholder="a name of its own" autocomplete="off"></label>
    <div class="row u-gap-8">
      <button id="dSaveNamed">save under this name</button>
      <button class="quiet" id="dSaveKeep">keep '${esc(name)}' and save</button>
    </div></div>`;
}

function wireNameAsk(box) {
  if (!box || !box.querySelector) return;
  const input = box.querySelector('#dSaveName');
  const named = box.querySelector('#dSaveNamed');
  const keep = box.querySelector('#dSaveKeep');
  if (named && input) named.onclick = () => saveUnderName(input.value);
  if (input) {
    input.onkeydown = event => {
      if (event.key === 'Enter') { event.preventDefault(); saveUnderName(input.value); }
    };
  }
  if (keep) keep.onclick = () => keepNameAndSave();
}

// The question in place of the save's answer, where that answer would have gone.
function askForAName(box, whose) {
  DRAFT_SAVED = nameAskHtml(whose);
  box.innerHTML = DRAFT_SAVED;
  wireNameAsk(box);
  showSaved(box);
  const input = box.querySelector && box.querySelector('#dSaveName');
  if (input && input.focus) input.focus();
  return null;
}

function keepNameAndSave() {
  NAME_KEPT = String(DRAFT.name || '').trim().toLowerCase();
  return saveDraft();
}

// The name typed in the question, written as the name field would write it, then saved.
// Nothing typed asks again rather than saving under the borrowed name.
function saveUnderName(typed) {
  const name = String(typed || '').trim();
  if (!name) {
    const input = document.getElementById('dSaveName');
    if (input && input.focus) input.focus();
    return null;
  }
  DRAFT.name = name;
  const field = document.getElementById('dName');
  if (field) field.value = name;
  keepDraft();
  return saveDraft();
}

function saveDraft() {
  const box = document.getElementById('dSaved');
  const whose = borrowedName();
  if (whose) return askForAName(box, whose);
  // Two handlers on the save's own promise rather than a `catch` over the whole chain:
  // what follows a save is the lineage and the rungs, and a fetch of those failing is
  // not the save being refused. It used to print one as the other.
  stampParent();
  return api('/api/design/save', { spec: DRAFT, parent: DRAFT_FROM }).then(
    kept => {
      // The library is keyed by hash and the hash covers mechanics only, so saving a
      // draft you only renamed is the game you already saved rather than a second one.
      DETAIL[kept.hash] = undefined;
      // Saved under it, the name is this game's own from now on.
      NAME_KEPT = String(DRAFT.name || '').trim().toLowerCase();
      DRAFT_SAVED = savedHtml(kept);
      box.innerHTML = DRAFT_SAVED;
      showSaved(box);
      askHint(kept.hash);
      // The saved game is the parent of whatever this draft becomes next, so the next
      // save extends the lineage rather than forking the original again.
      DRAFT_FROM = kept.hash;
      keepDraft();
      return api('/api/design/' + kept.hash).then(source => {
        DRAFT_PARENT = source;
        drawVerdict();
        return loadStanding(kept.hash);
      }).catch(fail);
    },
    error => {
      // A refused save is a 400 carrying a plain sentence -- the draft outran the
      // check's clock, or it is not a ruleset yet. It goes where the save's own
      // sentence would have gone, and it is the last thing the save said, so a redraw
      // of the panel keeps it.
      DRAFT_SAVED = `<div class="error">${esc(errorWords(error))}</div>`;
      box.innerHTML = DRAFT_SAVED;
      showSaved(box);
    });
}

// Ten moves of an idea, thrown away. The only honest answer to "is this rule
// interesting" is playing it, and until this existed the way to find out was to write
// the idea into the archive first -- so the loop that mattered most was closed by a
// permanent record of every experiment.
//
// The rules panel travels with the snapshot because the board's own tabs read the
// library by hash, and a draft is not in the library. It is the same `library.detail`
// every game page renders, built from the draft in the same request.
//
// `show` opens the board part-way into a game the engine played; see `showDraft`.
// `here` plays it on the editor's own board instead of opening the board page.
function playDraft(seats, show, here) {
  const box = document.getElementById('dSaved');
  return api('/api/design/play', { spec: DRAFT, seats, show })
    .then(snapshot => {
      if (snapshot.detail) DETAIL[snapshot.ruleset.hash] = snapshot.detail;
      if (here) return mountBenchGame(snapshot);
      open_(snapshot);
      noteWatching(snapshot);
    })
    .catch(error => {
      clearPending();
      const words = `<div class="error">${esc(errorWords(error))}</div>`;
      const mount = here ? document.getElementById('benchPlay') : null;
      if (mount) { mount.innerHTML = words + benchPlayHtml(); wireBenchPlay(); }
      else if (box) box.innerHTML = words;
      else fail(error);
    });
}

// The watch's budget as a person reads it, off the agent string, so the number in the
// sentence and the number in the seat cannot drift apart.
function watchNodes() {
  return Number((WATCH_AGENT.match(/nodes=([0-9]+)/) || [])[1] || 0).toLocaleString('en-US');
}

// `rounds`: the draft is played in rounds, where the watch seats the server's round twin of
// `WATCH_AGENT` (`opponents.seat_for`), whose budget is not this one, so no number is said.
function watchTitle(rounds) {
  if (rounds) {
    return 'two computer players for games in rounds, the strongest quick enough to watch;'
      + ' pause and step through it';
  }
  return `two computer players, each looking at ${watchNodes()} positions before each`
    + ' move; pause and step through it';
}

// Whether the draft's watch is a game in rounds from its first move. A spy game declares
// rounds but starts by turns, and its watch seats `WATCH_AGENT` itself.
function watchesRounds() {
  return !!DRAFT && turnWay() === 'rounds' && !(DRAFT.pieces || []).some(piece => piece && piece.spy === true);
}

// Who is playing a draft nobody is sitting at, in plain words.
//
// A watch is unattended, so the one question it raises -- *why are the moves like
// that* -- has to be answerable without leaving the board. Nothing said which player
// was playing, so a weak rung read as a broken game (Q265). The first sentence is the
// server's, which names the seat off the picker's own labels; the rest is the
// designer's, because the choice of player is the designer's.
//
// A game in rounds (Q998) offers nobody a move while two computers play it, so the count
// of legal moves read *0*; and the question its watch raises is a different one. Both
// players choose at once, from one position, each blind to the other and drawing from its
// own stream; but where the position is the same from both sides and one move is plainly
// best, the same player finds the same move from each side, and the game goes on mirrored
// until an ending reaches both players in one round, which is a draw (`rounds.md` §2.6).
// The user took that for a fault (*"both sides make identical moves"*), so the board says
// it. The watch seats the server's round twin of `WATCH_AGENT`, both seats alike.
function watchingLine(snapshot) {
  if (!snapshot || !snapshot.watching) return '';
  const seats = snapshot.seats || [];
  const rounds = !!snapshot.round;
  const watched = seats.length === 2 && seats.every(seat => seat === WATCH_AGENT)
    || rounds && seats.length === 2 && seats[0] === seats[1];
  return snapshot.watching
    + (watched ? ' It is the strongest computer player quick enough to watch a whole game.'
      : '')
    + (rounds ? roundWatchWords(seats[0] === seats[1])
      : ` This position offers ${(snapshot.moves || []).length} legal moves, and the more`
        + ' moves a position offers, the less far ahead any computer player sees.');
}

function roundWatchWords(same) {
  return ' Both players choose at once, from the same position, neither seeing the'
    + ' other\'s move.' + (same ? ' Where the position looks the same from both sides and'
    + ' one move is plainly best, the same player picks the same move for each side, so'
    + ' the game can run mirrored until both players reach an ending in the same round,'
    + ' which is a draw.' : '');
}

// The line, put on the board from here rather than drawn there: `page/board.js` draws a
// position for every game in the library and this is a fact about one draft's trial.
//
// `open_` navigates and the router draws the board on the hashchange, after fetching the
// roster that names the seats -- which on a cold page is the whole library listing. So the
// markup to write on is waited for rather than assumed, and the note is dropped if the
// game has been replaced in the meantime.
function noteWatching(snapshot) {
  const said = watchingLine(snapshot);
  if (!said) return;
  const until = Date.now() + 8000;
  const place = () => {
    if (!SESSION || SESSION.id !== snapshot.id) return;
    const split = view.querySelector('.split');
    // `#board` is the board's own svg. A game's page has a `.split` of its own, so the
    // two together are what says the board is the thing on screen.
    if (!split || !document.getElementById('board')) {
      if (Date.now() < until) setTimeout(place, 40);
      return;
    }
    if (document.getElementById('watching')) return;
    const note = document.createElement('div');
    note.className = 'muted small';
    note.id = 'watching';
    note.style.margin = '8px 0';
    note.textContent = said;
    split.insertAdjacentElement('beforebegin', note);
  };
  place();
}

// A game the engine played, opened where it matters: `{ rule }` stops just before the
// moves where that rule changed the game, `'unfinished'` at the end of a game that did
// not finish. Both seats are people, so nothing moves until somebody does.
function showDraft(show, message) {
  pending(message);
  return playDraft(['human', 'human'], show);
}

// ---------------------------------------------------------- building an opponent

let BOT = null;   // { hash, name, plan, weights, values, live, pieces }

async function showBotBuilder(hash) {
  // A server that keeps no opponents answers the builder's address with the game's own
  // page (D47). The address is not linked there; somebody who types it meets the game
  // rather than sliders whose result nothing can keep.
  if (!offered('POST', '/api/opponent')) {
    try { history.replaceState(null, '', gamePath(hash)); } catch (error) { /* stays */ }
    RENDERED = location.href;
    RENDERED_HASH = location.hash;
    return showGame(hash);
  }
  stopPolling();
  SESSION = null;
  FLIPPED = false;
  const [game, kit] = await Promise.all([detailFor(hash), designKit()]);
  if (!BOT || BOT.hash !== hash) {
    BOT = {
      hash,
      name: 'my opponent',
      plan: '',
      // Starts as something that plays: an all-zero vector is a random mover and is
      // refused, and a slider set nobody has touched should not be a refusal.
      weights: kit.vocabulary.map(term => term.key === 'material' ? 1 : 0),
      // One multiplier per piece, all at 1: agreeing with the rules about every piece
      // is the same opponent as having no opinion, which is what the server stores.
      values: (game.pieceValues || []).map(() => 1),
      live: game.liveTerms || [],
      pieces: game.pieceValues || [],
    };
  }

  view.innerHTML = `
    <button class="crumb" id="back">&larr; ${esc(game.name)}</button>
    <h2 class="u-mt-10">Build an opponent for ${esc(game.name)}</h2>
    <p class="muted small u-maxw-74ch">
      An opponent is a search, plus a list of what it is trying to achieve. All of them
      look at the same number of positions${info('thumb')}, so what you set here changes
      what it <i>wants</i> and nothing else. Push a slider up to make it chase something,
      down to make it avoid it &mdash; giving material away is a real plan in some games.
    </p>
    <div class="designer u-mt-12">
      <div class="panel">
        <div class="pair">
          <label class="field"><span>what to call it</span>
            <input id="botName" value="${esc(BOT.name)}"></label>
          <label class="field"><span>what it is trying to do, in a sentence</span>
            <input id="botPlan" value="${esc(BOT.plan)}"
              placeholder="Win material and keep the middle."></label>
        </div>
        <div class="sliders u-mt-12">${
          kit.vocabulary.map((term, index) => sliderHtml(term, index)).join('')}</div>
      </div>
      ${piecePanelHtml(kit)}
      <div>
        <div class="panel">
          <h3 class="u-mt-0">What it will play for</h3>
          <div id="botSummary"></div>
          <div class="row u-mt-10">
            <button class="primary" id="botSave">save it</button>
            <button id="botPlay">save and play it</button>
          </div>
          <div id="botResult"></div>
        </div>
        <div class="panel u-mt-14">
          <h3 class="u-mt-0">Already built</h3>
          <div id="botList">${builtHtml(game)}</div>
        </div>
      </div>
    </div>`;

  document.getElementById('back').onclick = () => go('#/game/' + hash);
  wireBot(kit);
  drawBotSummary(kit);
}

function sliderHtml(term, index) {
  const inert = BOT.live.length && !BOT.live.includes(term.key);
  const value = BOT.weights[index] || 0;
  return `<div class="slider u-op-0_45" title="${esc(term.gloss)}"${inert ? '' : ''}>
    <span>${esc(term.key.replace(/_/g, ' '))}${inert
      ? `<br><span class="muted u-fs-11">needs ${esc(term.needs)}</span>` : ''}</span>
    <input type="range" data-term="${index}" min="-2" max="2" step="0.05" value="${value}"${
      inert ? ' disabled' : ''}>
    <span class="val" data-val="${index}">${value.toFixed(2)}</span>
  </div>`;
}

// The second half of an opponent: what it thinks the pieces are worth.
//
// Shown as a multiplier of the value the rules imply rather than as a number, because
// the derived table spans 1.25 to 74 on xiangqi and a person cannot hold that scale in
// their head -- "the elephant is worth five times what the engine thinks" is a statement
// they can make and check. Log scale for the same reason: half and double should be the
// same distance from one.
function piecePanelHtml(kit) {
  if (!BOT.pieces.length) return '';
  return `
    <div class="panel u-gcol-1-n1 u-mt-14">
      <h3 class="u-mt-0">What the pieces are worth to it</h3>
      <p class="muted small u-maxw-74ch">
        Each slider is a multiple of what the rules already imply, worked out from how
        much of the board the piece can reach. Moving one <i>shifts worth</i> between the
        pieces, because the total stays the same. Pushing every slider up together is the
        same opponent as leaving them all at 1&times;. Push one up and the others are
        worth less by comparison, which is the only thing "worth more" can mean.
      </p>
      <div class="sliders u-mt-12">${
        BOT.pieces.map((piece, index) => pieceSliderHtml(piece, index)).join('')}</div>
    </div>`;
}

function pieceSliderHtml(piece, index) {
  const multiplier = BOT.values[index] || 1;
  const icon = piece.icon
    ? `<img src="${esc(piece.icon)}" alt="" class="u-w-18 u-h-18 u-va-n4">`
    : '';
  return `<div class="slider u-op-0_45"${piece.royal ? '' : ''}>
    <span>${icon} ${esc(piece.name)}${piece.royal
      ? `<br><span class="muted u-fs-11">${esc(piece.needs)}</span>`
      : `<br><span class="muted u-fs-11">the rules say ${
          piece.derived.toFixed(2)}</span>`}</span>
    <input type="range" data-piece="${index}" min="-1" max="1" step="0.02"
      value="${Math.log10(multiplier)}"${piece.royal ? ' disabled' : ''}>
    <span class="val" data-piecev="${index}">${
      piece.royal ? '--' : multiplier.toFixed(2) + '×'}</span>
  </div>`;
}

function wireBot(kit) {
  document.getElementById('botName').oninput = e => { BOT.name = e.target.value; };
  document.getElementById('botPlan').oninput = e => { BOT.plan = e.target.value; };
  view.querySelectorAll('[data-term]').forEach(node => {
    node.oninput = () => {
      const index = Number(node.dataset.term);
      BOT.weights[index] = Number(node.value);
      const shown = view.querySelector(`[data-val="${index}"]`);
      if (shown) shown.textContent = BOT.weights[index].toFixed(2);
      drawBotSummary(kit);
    };
  });
  view.querySelectorAll('[data-piece]').forEach(node => {
    node.oninput = () => {
      const index = Number(node.dataset.piece);
      BOT.values[index] = Math.pow(10, Number(node.value));
      const shown = view.querySelector(`[data-piecev="${index}"]`);
      if (shown) shown.textContent = BOT.values[index].toFixed(2) + '×';
      drawBotSummary(kit);
    };
  });
  const save = document.getElementById('botSave'), play = document.getElementById('botPlay');
  if (save) save.onclick = () => saveBot(kit, false);
  if (play) play.onclick = () => saveBot(kit, true);
}

// The vector said back as a sentence, which is the test of whether it is a plan at all.
// A weight vector nobody can state is a fit rather than an idea, and that distinction is
// the whole reason a heuristic carries English.
function drawBotSummary(kit) {
  const box = document.getElementById('botSummary');
  if (!box) return;
  const ranked = BOT.weights
    .map((weight, index) => ({ weight, term: kit.vocabulary[index] }))
    .filter(entry => Math.abs(entry.weight) > 0.02)
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  if (!ranked.length) {
    box.innerHTML = `<div class="muted small">Nothing is set, so it would move at
      random. Push at least one slider off zero.</div>`;
    return;
  }
  box.innerHTML = `<ul class="rules small u-m-0">${ranked.map(entry => `
    <li><b>${entry.weight > 0 ? 'more' : 'less'}
      ${esc(entry.term.key.replace(/_/g, ' '))}</b>
      <span class="muted">${esc(entry.term.gloss)}</span></li>`).join('')}</ul>
    <div class="muted small u-mt-8">Only the balance between these
      matters, so doubling every slider makes the same opponent.</div>
    ${valueSummaryHtml()}`;
}

// The valuation said back in English, for the reason the weights are: a vector nobody
// can state is a fit rather than an idea, and the difference is the whole reason a
// heuristic carries a sentence at all.
function valueSummaryHtml() {
  const moved = BOT.pieces
    .map((piece, index) => ({ piece, multiplier: BOT.values[index] || 1 }))
    .filter(entry => !entry.piece.royal && Math.abs(Math.log10(entry.multiplier)) > 0.02);
  if (!moved.length) {
    if (!BOT.pieces.length) return '';
    return `<div class="muted small u-mt-10">It values the pieces exactly
      as the rules imply.</div>`;
  }
  // Everything moved by the same amount is a reallocation of nothing, and the server
  // stores no valuation at all for it. Say so here rather than let somebody discover it.
  const spread = new Set(moved.map(e => e.multiplier.toFixed(2)));
  if (moved.length === BOT.pieces.filter(p => !p.royal).length && spread.size === 1) {
    return `<div class="muted small u-mt-10">Every piece moved by the
      same amount, which changes nothing. It values them exactly as the rules imply.</div>`;
  }
  const ranked = moved.sort(
    (a, b) => Math.abs(Math.log10(b.multiplier)) - Math.abs(Math.log10(a.multiplier)));
  return `<ul class="rules small u-m-10-0-0">${ranked.map(entry => `
    <li><b>${esc(entry.piece.name)} ${entry.multiplier > 1 ? 'worth more' : 'worth less'}</b>
      <span class="muted">${entry.multiplier.toFixed(2)}&times; what the rules imply,
      which is ${entry.piece.derived.toFixed(2)}</span></li>`).join('')}</ul>`;
}

// Which store an opponent came from, not which menu heading it sits under: `group` is a
// label and gets reworded, and a list that emptied because somebody improved a heading
// would be a silent failure.
function builtHtml(game) {
  const mine = (game.opponents || []).filter(o => o.store === 'custom');
  if (!mine.length) {
    return `<div class="muted small">Nothing is built yet. What you build here is saved beside the
      game and shows up in the opponent list on its page.</div>`;
  }
  return `<ul class="rules small u-m-0">${mine.map(o =>
    `<li><b>${esc(o.label)}</b> <span class="muted">${esc(o.plan)}</span></li>`).join('')}</ul>`;
}

function saveBot(kit, andPlay) {
  const box = document.getElementById('botResult');
  api('/api/opponent', {
    hash: BOT.hash, name: BOT.name, plan: BOT.plan, weights: BOT.weights,
    values: BOT.values,
  }).then(built => {
    // The game's detail carries the roster, and the roster just grew.
    delete DETAIL[BOT.hash];
    if (andPlay) {
      return api('/api/session', { hash: BOT.hash, seats: ['human', built.agent] })
        .then(open_);
    }
    box.innerHTML = `<div class="ok u-mt-10"><b>${esc(built.name)}</b> is
      ready, and is now in the opponent list for this game.</div>`;
    detailFor(BOT.hash).then(game => {
      const list = document.getElementById('botList');
      if (list) list.innerHTML = builtHtml(game);
    });
  }).catch(error => {
    box.innerHTML = `<div class="error">${esc(errorWords(error))}</div>`;
  });
}
