// A game's proven positions, on a page of their own (Q993b, `docs/plan-proven.md` §2):
// `/game/<hash>/proven`, and `/game/<hash>/proven/<id>` with the position chosen, which the
// router also reads as `#/proven/<hash>[/<id>]`. The user, 2026-10-20: *"Better UX for
// seeing progress, choosing challenges, seeing which are done. work through them without
// going back and forth to the game site."*
//
// The list beside the board (the frame first on a phone, the list under it), the progress
// line and the done marks over it, *previous* and *next* under the board, and a try played
// in this frame, with *next position* on its ended banner. Everything here is the server's
// (`site/proven.py`): the list and its order, the groups, the chips' counts, the drawing,
// the moves to pick from (never their scores), every answer and every sentence. The page
// sorts nothing and judges nothing (rule 3); it narrows the server's list in place and walks
// it. A try is the core's practice game against hardest, drawn by the one playing board
// (`showBoard`, its third chrome), which its own ending decides.
//
// A visitor's right picks are ticked for the visit: held here, in the page, and sent back
// with the list so the server counts them, never kept in the browser or anywhere else.

let PROVEN = null;           // { hash, data, id, playing }: the view on screen
const PROVEN_VISIT = {};     // hash -> the ids a visitor found this visit
const PROVEN_CHIPS = {};     // chip key -> on, for the visit

function provenPath(hash, id) {
  return '/game/' + hash + '/proven' + (id ? '/' + encodeURIComponent(id) : '');
}

// The side to move by the game's own name for it (*black*, *south*), else by its place.
function provenSide(sides, side) {
  const named = (sides || [])[side];
  const said = named || (side === 0 ? 'the first player' : 'the second player');
  return said.charAt(0).toUpperCase() + said.slice(1);
}

// The still board from the side to move, as the board a try opens on is drawn: the one
// flip the reader's own board would make, put back as soon as the drawing is made.
function provenBoard(preview, side) {
  const was = FLIPPED;
  FLIPPED = side === 1;
  try { return boardPreview(preview, null); } finally { FLIPPED = was; }
}

function provenAsk(hash) {
  const found = [...(PROVEN_VISIT[hash] || [])];
  return api('/api/proven/' + hash + (found.length
    ? '?found=' + found.map(encodeURIComponent).join(',') : ''));
}

async function showProven(hash, id) {
  stopPolling();
  if (SESSION) leaveBoard();
  SESSION = null;
  SHOWN = null;
  if (!ACCOUNTS) {
    view.innerHTML = `<div class="panel"><h2 style="margin-top:0">Proven positions</h2>
      <div class="muted">Proven positions are offered on the site, where a try is played
        against hardest. This application offers none.</div></div>`;
    return;
  }
  // A step along the list redraws the view from the list in hand; any other way in asks.
  const same = !!(PROVEN && PROVEN.hash === hash && document.getElementById('provenList'));
  // The server drew these positions into the view for a reader without script (G10,
  // `proven_page.py`); the page draws its own, so that goes before the wait.
  if (!same) { view.innerHTML = ''; pending('Reading the positions…'); }
  const data = same ? PROVEN.data : await provenAsk(hash);
  clearPending();
  const ids = data.positions.map(entry => entry.id);
  const at = ids.includes(id) ? id : ids[0] || '';
  PROVEN = { hash, data, id: at, playing: null };
  setTitle(`Proven positions on ${data.name || 'an unnamed game'}`);
  // An address naming a position the list does not offer (never, or no longer: G10, the
  // server answered it 404 with the list) opens at the list's first, and says why.
  const gone = !!(id && at !== id);
  if (gone) {
    try { history.replaceState(null, '', provenPath(hash, at)); } catch (error) { /* stays */ }
  }
  view.innerHTML = provenViewHtml(data, gone);
  wireProvenView();
  if (at) showProvenAt(at);
}

function provenViewHtml(data, gone) {
  const name = esc(data.name || 'an unnamed game');
  const words = data.words || {};
  const frame = data.positions.length
    ? `<div class="gamePlay"><div class="panel proven-frame" id="provenMount"></div></div>
       <div class="gameSide"><div class="panel proven-list" id="provenList">${provenListHtml()}</div></div>`
    : `<div class="gamePlay"><div class="panel"><p id="provenNone">${esc(data.none)}</p></div></div>`;
  return `
    <a class="crumb" id="back" href="/game/${esc(PROVEN.hash)}">&larr; ${name}</a>
    <h2 style="margin:10px 0 0">Proven positions on ${name}${info('provenPositions')}</h2>
    ${gone ? `<p class="muted small" id="provenGone">${esc(words.gone || '')}</p>` : ''}
    <div class="gameTop provenTop">${frame}</div>
    <details class="muted small" style="margin-top:12px">
      <summary>how these were proved</summary>
      <p id="provenBasis">${esc(data.basis)}</p>
    </details>`;
}

// The list: the progress line over it, the chips, then a group a length of win, each row
// a link to its position. The current row is marked neutral, as the move list's is.
function provenListHtml() {
  const data = PROVEN.data;
  const words = data.words || {};
  const phone = window.innerWidth <= 860;
  const current = PROVEN.id;
  const groups = data.groups.map(group => {
    const rows = group.ids.map(id => data.positions.find(entry => entry.id === id))
      .filter(Boolean);
    const open = !phone || data.groups.length === 1 || group.ids.includes(current);
    return `<details class="proven-group" data-ends="${group.endsIn}"${open ? ' open' : ''}>
      <summary>${esc(group.heading)}</summary>
      <ol class="proven-rows">${rows.map(provenRowHtml).join('')}</ol></details>`;
  }).join('');
  const chips = (data.chips || []).map(chip => `<button class="chip${
    PROVEN_CHIPS[chip.key] ? ' on' : ''}" data-chip="${esc(chip.key)}" aria-pressed="${
    PROVEN_CHIPS[chip.key] ? 'true' : 'false'}">${esc(chip.label)} · ${chip.count}</button>`).join('');
  return `
    ${data.progress ? `<p class="small" id="provenProgress" style="margin-top:0">${esc(data.progress)}</p>` : ''}
    ${data.keep ? `<p class="muted small" id="provenKeep"${data.progress ? '' : ' style="margin-top:0"'}>${esc(data.keep)}</p>` : ''}
    ${chips ? `<div class="chips" role="group" aria-label="narrow the list">${chips}</div>` : ''}
    <div id="provenGroups">${groups}</div>
    <p class="muted small" id="provenEmpty" hidden>${esc(words.empty || '')}</p>`;
}

function provenRowHtml(entry) {
  const words = PROVEN.data.words || {};
  const here = entry.id === PROVEN.id;
  const playing = PROVEN.playing === entry.id;
  const done = playing ? words.playing : entry.done;
  return `<li data-row="${esc(entry.id)}"${provenShown(entry) ? '' : ' hidden'}>
    <a class="proven-row${here ? ' on' : ''}" href="${provenPath(PROVEN.hash, entry.id)}"${
      here ? ' aria-current="true"' : ''}>
      <span class="n">${entry.number}</span>
      <span class="what">${esc(provenSide(PROVEN.data.sides, entry.side))} to move${
        entry.single ? `<span class="muted small">${esc(words.single)}</span>` : ''}</span>
      <span class="done">${done ? `${playing ? '' : '<span aria-hidden="true">&#10003;</span> '}${esc(done)}` : ''}</span>
    </a></li>`;
}

// Whether a row passes the chips that are on: a narrowing of the server's list, by the
// fields the server sent with each row.
function provenShown(entry) {
  if (PROVEN_CHIPS.unsolved && entry.solved) return false;
  if (PROVEN_CHIPS.single && !entry.single) return false;
  return true;
}

// The position `by` steps along the list as narrowed, from the one on screen, round the
// end; null where no other position passes the chips.
function provenStep(by) {
  const rows = PROVEN.data.positions;
  const here = rows.findIndex(entry => entry.id === PROVEN.id);
  for (let step = 1; step <= rows.length; step += 1) {
    const entry = rows[(here + by * step + rows.length * step) % rows.length];
    if (entry && entry.id !== PROVEN.id && provenShown(entry)) return entry.id;
  }
  return null;
}

function redrawProvenList() {
  const list = document.getElementById('provenList');
  if (!list || !PROVEN) return;
  list.innerHTML = provenListHtml();
  wireProvenList();
  const shown = PROVEN.data.positions.some(provenShown);
  const empty = document.getElementById('provenEmpty');
  if (empty) empty.hidden = shown;
  document.querySelectorAll('.proven-group').forEach(group => {
    group.hidden = !group.querySelector('li:not([hidden])');
  });
  provenStepsShown();
}

function wireProvenView() {
  wireProvenList();
  const list = document.getElementById('provenList');
  // A row is a link (keyboard and screen readers walk the list with no script of their own)
  // and is followed in place by the router's link handler. A try in progress asks first.
  if (list) list.addEventListener('click', event => {
    const link = event.target.closest && event.target.closest('a.proven-row');
    if (link && !provenMayLeave()) event.preventDefault();
  }, true);
}

function wireProvenList() {
  document.querySelectorAll('#provenList [data-chip]').forEach(chip => {
    chip.onclick = () => {
      PROVEN_CHIPS[chip.dataset.chip] = !PROVEN_CHIPS[chip.dataset.chip];
      redrawProvenList();
    };
  });
  document.querySelectorAll('.proven-group').forEach(group => {
    group.hidden = !group.querySelector('li:not([hidden])');
  });
  const empty = document.getElementById('provenEmpty');
  if (empty) empty.hidden = PROVEN.data.positions.some(provenShown);
}

// A try that has not ended is a game that will be left unfinished: asked plainly first.
function provenMayLeave() {
  if (!SESSION || !SESSION.puzzle || SESSION.outcome) return true;
  const words = (PROVEN && PROVEN.data.words) || {};
  if (!window.confirm(words.leave || 'Leave this game unfinished?')) return false;
  leaveBoard();
  return true;
}

// To another position: the address names it, and the router draws it in place.
function provenGo(id) {
  if (!id || !provenMayLeave()) return;
  go('#/proven/' + PROVEN.hash + '/' + id);
}

// *previous* and *next* name what they do, and say so when nothing else passes the chips.
function provenStepsShown() {
  const none = provenStep(1) === null;
  ['provenPrev', 'provenNext'].forEach(key => {
    const button = document.getElementById(key);
    if (button) {
      button.disabled = none;
      button.title = none ? (PROVEN.data.words.alone || '') : button.dataset.title || '';
    }
  });
}

// ------------------------------------------------------------------ the position

function showProvenAt(id) {
  const mount = document.getElementById('provenMount');
  if (!mount) return;
  mount.innerHTML = '<div class="muted small">Drawing the position…</div>';
  api('/api/proven/' + PROVEN.hash + '/' + encodeURIComponent(id))
    .then(position => {
      if (!PROVEN || PROVEN.id !== id) return;
      // G10: the tab says what the position's own address says (`proven.position_title`).
      if (position.title) setTitle(position.title);
      mount.innerHTML = provenFrameHtml(position);
      wireProvenFrame(position);
      fitProvenStill();
    })
    .catch(error => { mount.innerHTML = ''; fail(error); });
}

// The frame (plan-visual.md §3.3): the status strip above the board (S1), the board sized
// to fit (S2), the control bar under it with *previous*, *next* and the view's one primary
// (S6), and, where one move wins, *find the move* under the bar.
function provenFrameHtml(position) {
  const data = PROVEN.data;
  const words = data.words || {};
  const preview = Object.assign({}, position.preview, { hash: PROVEN.hash });
  const play = data.signInToConvert
    ? `<span class="muted small" id="provenSignIn">${esc(data.signInToConvert)}</span>`
    : `<button class="primary" id="provenPlay"
        title="a practice game from this position against hardest; you convert it if the game ends in your win"
        >${esc(words.play)}</button>`;
  const find = position.single && position.moves.length ? `
    <h4 style="margin:14px 0 4px">${esc(words.find)}</h4>
    <div class="muted small">${esc(words.findSaid)}</div>
    <div class="row proven-moves" style="margin-top:6px">${position.moves.map(entry =>
      `<button class="quiet" data-proven-move="${entry.move}">${esc(entry.text)}</button>`).join('')}
    </div>
    <div id="provenSaid" class="small" style="margin-top:6px" aria-live="polite"></div>
    <div class="row" style="margin-top:6px">
      <button id="provenShow" class="quiet" title="say which move the computer proved, without picking one"
        >${esc(words.show)}</button>
    </div>` : '';
  return `
    <div class="row frame-status" data-frame="status"><span id="provenStatus">${
      esc(provenSide(data.sides, position.side))} to move. ${esc(position.said)}</span></div>
    <div class="proven-still" data-frame="board" role="img"
      aria-label="Proven position ${position.number}." data-tall="${
        (preview.height / Math.max(1, preview.width)).toFixed(4)}">${provenBoard(preview, position.side)}</div>
    <div class="row frame-controls" data-frame="controls">
      <button class="quiet" id="provenPrev" data-title="the position before this one on the list"
        title="the position before this one on the list">${esc(words.previous)}</button>
      <button class="quiet" id="provenNext" data-title="the position after this one on the list"
        title="the position after this one on the list">${esc(words.nextShort || 'next')}</button>
      ${play}
      ${provenCopyHtml(position)}
    </div>
    ${provenLegendHtml(preview)}
    ${find}`;
}

// G10: the position's own address, to send to somebody: the one copy control the page has
// (`core.js` `wireCopyLinks`), naming the game and the position, so what it copies is the
// address the server draws the position at, its canonical. No share prompt, no count.
function provenCopyHtml(position) {
  const words = (PROVEN && PROVEN.data.words) || {};
  return `<button class="quiet" id="provenCopy" data-copy-link="${esc(PROVEN.hash)}"
    data-copy-position="${esc(position.id)}" title="${esc(words.copyTitle || '')}"
    >${esc(words.copy || 'copy link')}</button>`;
}

// The legend line under the still board (S3): every kind of mark it draws, named, as the
// game page's start position names its own (`openingHtml`).
function provenLegendHtml(preview) {
  const kinds = new Set();
  const marks = squareMarks(preview, { zone: !zonesShown() });
  for (const square of Object.keys(marks)) {
    for (const mark of marks[square]) kinds.add(mark.kind);
  }
  boardKinds(preview.obstacles, preview.portals).forEach(kind => kinds.add(kind));
  if (walksDrawn(preview)) kinds.add('walk');
  const said = legendEntries(kinds, preview.wrap, '', preview.portals);
  return said.length ? `<div class="frame-legend small" data-frame="legend">on the board: ${
    said.join(' ')}</div>` : '';
}

// The still board's side, so the strip, the board and the bar are all in the first screen
// (S2): never wider than its column, never so tall that the bar under it falls off.
function fitProvenStill() {
  const still = document.querySelector('.proven-still');
  const bar = document.querySelector('.proven-frame [data-frame="controls"]');
  if (!still || !bar || typeof still.getBoundingClientRect !== 'function') return;
  const legend = document.querySelector('.proven-frame [data-frame="legend"]');
  const tall = Number(still.dataset.tall) || 1;
  still.style.maxWidth = '';
  const box = still.getBoundingClientRect();
  // Under the board: the bar and the legend line, held to the first screen with it.
  const under = Math.max(bar.getBoundingClientRect().bottom,
    legend ? legend.getBoundingClientRect().bottom : 0) - box.bottom;
  const room = window.innerHeight - (box.top + (window.scrollY || 0)) - under - 16;
  const side = Math.max(BOARD_FLOOR, room);
  still.style.maxWidth = Math.floor(side / tall) + 'px';
}

function wireProvenFrame(position) {
  const words = PROVEN.data.words || {};
  const prev = document.getElementById('provenPrev');
  const next = document.getElementById('provenNext');
  if (prev) prev.onclick = () => provenGo(provenStep(-1));
  if (next) next.onclick = () => provenGo(provenStep(1));
  provenStepsShown();
  const play = document.getElementById('provenPlay');
  if (play) play.onclick = () => provenTry(position);
  const said = document.getElementById('provenSaid');
  const buttons = Array.from(document.querySelectorAll('[data-proven-move]'));
  const ask = body => api('/api/proven/' + PROVEN.hash + '/' + encodeURIComponent(position.id) + '/find', body)
    .then(answer => {
      said.textContent = answer.said + (answer.marked ? ' ' + (words.marked || '') : '');
      said.className = 'small ' + (answer.right ? 'ok' : 'muted');
      if (answer.right || answer.shown) buttons.forEach(button => { button.disabled = true; });
      if (answer.right) provenSolved(position.id, answer.marked ? 'found' : 'visit');
    })
    .catch(error => fail(error, said));
  buttons.forEach(button => {
    button.onclick = () => ask({ move: Number(button.dataset.provenMove) });
  });
  const reveal = document.getElementById('provenShow');
  if (reveal) reveal.onclick = () => ask({ show: true });
  if (typeof window.addEventListener === 'function' && !wireProvenFrame.wired) {
    wireProvenFrame.wired = true;
    window.addEventListener('resize', () => { if (!SESSION) fitProvenStill(); });
  }
}

// A position solved on this page: its row ticked at once from the answer, then the list
// asked again, so the progress line and the chips' counts are the server's. A visitor's
// find is held for the visit and sent back with the list; nothing is kept for them.
function provenSolved(id, how) {
  if (!PROVEN) return;
  const hash = PROVEN.hash;
  const entry = PROVEN.data.positions.find(row => row.id === id);
  if (how === 'visit') {
    (PROVEN_VISIT[hash] = PROVEN_VISIT[hash] || new Set()).add(id);
  }
  if (entry) {
    if (how === 'converted') entry.converted = true;
    else entry.found = true;
    entry.solved = true;
    const words = PROVEN.data.words || {};
    entry.done = how === 'visit' ? words.visit
      : entry.converted && entry.found ? words.both
        : entry.converted ? words.converted : words.found;
  }
  redrawProvenList();
  provenAsk(hash).then(data => {
    if (!PROVEN || PROVEN.hash !== hash) return;
    PROVEN.data = data;
    redrawProvenList();
  }).catch(() => {});
}

// ------------------------------------------------------------------ a try, in place

function provenTry(position) {
  const hash = PROVEN.hash;
  pending('Setting up the position…');
  api('/api/session', { hash, puzzle: position.id })
    .then(snapshot => provenPlay(snapshot))
    .catch(error => fail(error, 'provenPlay'));
}

// The try drawn into this view's mount by the one playing board, in its third chrome: the
// list stays beside it with the row marked *playing*, and its ended banner offers *next
// position* first and *try it again* beside it.
function provenPlay(snapshot) {
  const mount = document.getElementById('provenMount');
  if (!mount || !PROVEN) { open_(snapshot); return; }
  clearPending();
  PROVEN.playing = PROVEN.id;
  redrawProvenList();
  const words = PROVEN.data.words || {};
  return showBoard(snapshot, {
    mount,
    proven: {
      words,
      previous: () => provenGo(provenStep(-1)),
      next: () => provenGo(provenStep(1)),
      again: next => provenPlay(next),
      moved: provenMoved,
    },
  }).then(provenStepsShown);
}

// Every snapshot of the try passes here. At its end the row takes its tick from the reply
// (`puzzle.converted`, the mark the finish wrote), with no reload.
function provenMoved(snapshot) {
  if (!PROVEN || !snapshot.puzzle || !snapshot.outcome) return;
  if (PROVEN.playing === snapshot.puzzle.id) PROVEN.playing = null;
  if (snapshot.puzzle.converted) provenSolved(snapshot.puzzle.id, 'converted');
  else redrawProvenList();
}
