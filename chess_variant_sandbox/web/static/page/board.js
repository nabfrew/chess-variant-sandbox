// The board: drawing a position, animating a move, and playing one.

// ------------------------------------------------------------------ playing

// Where this board is drawn. The play page gives it the whole view; the designer gives
// it one panel of the column it edits in. Everything below this line is the same board
// either way -- the markup carries the same ids, and `document.getElementById` does not
// care which of the two wrote them.
//
// What the two do differ in is chrome. A page board is the only thing on screen, so it
// carries the crumb back to the game and the panel of rules and pieces beside it; an
// inline board sits beside the editor that already shows both, so it carries neither and
// gives that width to the board.
let BOARD_INLINE = false;

// The board, given up. Everything it holds is about one game, so a view that takes the
// board away drops all of it: a reply still on its way lands on a session nobody is
// showing and is discarded by `maybeAgent`'s own check, and the next game opens without
// the last one's timeline behind it.
function leaveBoard() {
  if (BOARD_SIGHT) BOARD_SIGHT.disconnect();
  BOARD_SIGHT = null;
  BOARD_OFFSCREEN = false;
  SESSION = null;
  SHOWN = null;
  SELECTED = null;
  DROP = null;
  HIGHLIGHT = null;
  PHASE = null;
  NOTE = '';
  AT = null;
  CURSOR = null;
  PAUSED = false;
  TIMELINE = [];
  TIMELINE_FOR = null;
  BOARD_INLINE = false;
  CORR_HELD = null;
  stopCorrPoll();
}

// `how` is `{ mount }` for a board drawn inside another view, and absent for the page's
// own board.
async function showBoard(snapshot, how) {
  stopPolling();
  stopCorrPoll();
  CORR_HELD = null;
  const inline = !!(how && how.mount);
  BOARD_INLINE = inline;
  const root = inline ? how.mount : view;
  SESSION = snapshot;
  SHOWN = null;
  SELECTED = null;
  DROP = null;
  HIGHLIGHT = null;
  PHASE = null;
  NOTE = '';
  // A first game opens on the rules (K3); `OPEN_TAB` is taken once.
  TAB = OPEN_TAB || 'moves';
  OPEN_TAB = null;
  const rules = snapshot.ruleset;
  // A deep link to a game in progress never visited the picker, so the roster that
  // names the seats has to be fetched here as well. Without it a rule of thumb is
  // labelled with its own digest. A draft has no page to ask, so it takes the ladder
  // everything shares rather than a roster nobody can look up.
  await roster(rules.scratch ? '' : rules.hash).catch(() => {});
  // Your own side at the near end, which is the only orientation a person can read a
  // board in. A game with nobody in it, or with two people in it, keeps the first
  // player at the bottom; the flip button is there for both.
  FLIPPED = snapshot.seats[1] === 'human' && snapshot.seats[0] !== 'human';
  // A game between two people (Q570) is drawn from the reader's own side.
  const between = snapshot.correspondence || null;
  if (between) FLIPPED = between.you === 1;
  // A board opened starts in the frame its squares are named in. The shift is a way of
  // reading one board, not a preference carried between boards -- two games rarely have
  // the same width, and a remembered offset would land somewhere arbitrary.
  SHIFT = { files: 0, ranks: 0 };
  const mixed = snapshot.seats[0] !== snapshot.seats[1];
  // Nobody is at this board: both sides are the machine's and the game is there to be
  // read rather than played. It is what the watch control starts, and it is why that
  // control is offered on every other board and not on this one.
  const watched = !snapshot.seats.includes('human');
  // Whether this game counts is `snapshot.mode`, as the store says (`site/matches.py`),
  // and `frameHtml` reads it: a rated game has no take-back, a sandbox game has one, and
  // both can be resigned; the local application has no modes at all and sends nothing,
  // so it gets no mark and no resignation and keeps its take-back.
  //
  // The editor's trial says the *game* is not recorded, never that the rules are not
  // saved: they may well be, and after a save that read as a contradiction (Q576, N4).

  root.innerHTML = `
    ${inline ? `<div class="muted small">This is a game of the rules as they stand right
      now. Nothing about it is kept, and changing the draft ends it.</div>`
      : `<button class="crumb" id="back">&larr; ${
        rules.scratch ? 'back to the editor' : esc(rules.name)}</button>`}
    ${!inline && rules.scratch ? `<div class="muted small" style="margin-top:8px">
      This is a trial of <b>${esc(rules.name)}</b>, and this trial isn't recorded: nothing
      about it is kept, and going back to the editor leaves no trace of it.</div>` : ''}
    ${snapshot.demo ? `<div class="ok" style="margin-top:8px">${esc(snapshot.demo.says)}
      The computer played these moves. Step forward to watch them, or move from here.</div>` : ''}
    <div class="split${inline ? ' alone' : ''}" style="margin-top:10px">
      <div class="play-frame">
        ${inline ? `<div class="row small" style="margin-bottom:8px">
          <div>${seatHtml(0)}</div><span class="muted">vs</span><div>${seatHtml(1)}</div>
        </div>` : ''}
        ${frameHtml(inline, snapshot.mode, watched, between)}
        <div class="row" id="review" style="margin-top:8px"></div>
        ${between ? '' : `<div class="row" style="margin-top:8px">
          <button id="again">new game</button>
          ${mixed ? `<button id="swap" title="the same game from the other side">
            play the other side</button>` : ''}
          ${watched ? '' : `<button id="watch" title="${esc(watchTitle())}">
            watch it play itself</button>`}
        </div>`}
        <div id="ground" class="muted small" style="margin-top:6px"></div>
      </div>
      ${inline ? '' : `<div class="panel">
        <div>${seatHtml(0)}</div>
        <div style="margin-top:6px">${seatHtml(1)}</div>
        ${modeHtml(snapshot)}
        <div class="tabs" style="margin-top:10px">
          ${['moves', 'rules', 'pieces'].map(tab => `<button data-tab="${tab}"${
            tab === TAB ? ' class="on"' : ''}>${tab}</button>`).join('')}
        </div>
        <div class="tabpane" id="pane"></div>
      </div>`}
    </div>`;

  // A draft is not in the library, so every one of these three goes through the editor's
  // own scratch route instead of through a hash nothing can resolve.
  //
  // `pending` fires here rather than in each button, so both of them -- new game and
  // play the other side -- show it exactly once, before the one slow call either way.
  //
  // This is the rematch control, so it names the game it was pressed on and the store
  // records the new match as that one's rematch (`matches.rematch_of`). Starting from a
  // game page or from the editor sends nothing, and those rows stay null. Only the match
  // id goes, because the server already knows who is asking and the match it names
  // already holds both seats.
  //
  // A game nobody sat at is nobody's to play again, and neither is a watch started from
  // one, so both leave the column null and the store refuses either as a rematch.
  //
  // The new game is played in the mode this one was. Sending nothing meant the server's
  // default, so a rematch of a rated game was a sandbox game and the board silently
  // stopped counting (Q249). A game nobody sits at is the exception and sends no mode:
  // nobody plays it, so there is nothing to rate, and the store refuses a rated watch.
  const restart = (seats, follows = true) => {
    // An inline board is always a draft's, and it restarts where it stands: the editor
    // puts the spinner in the board's own panel, because the page's own is at the top of
    // a form that is several screens tall.
    if (inline) return playDraftHere(seats);
    pending('Starting the game…');
    if (rules.scratch) return playDraft(seats);
    const body = { hash: rules.hash, seats };
    if (snapshot.mode && seats.includes('human')) body.mode = snapshot.mode;
    if (follows && !watched) body.rematch_of = snapshot.id;
    return api('/api/session', body).then(open_).catch(fail);
  };
  // Never by presence: the designer has a crumb of its own called `back`, and an inline
  // board that looked one up would rewire the editor's way out of the page.
  if (inline) document.getElementById('stopPlay').onclick = () => stopPlayingHere();
  else document.getElementById('back').onclick = () =>
    go(rules.scratch ? '#/design' : '#/game/' + rules.hash);
  const again = document.getElementById('again');
  if (again) again.onclick = () => restart(snapshot.seats);
  // A rematch with the seats exchanged, which is the whole of "let me try that as the
  // other side" and is a new game rather than a redraw: the two armies are different
  // pieces and the position is the opening one either way.
  if (mixed && !between) document.getElementById('swap').onclick = () =>
    restart([snapshot.seats[1], snapshot.seats[0]]);
  // Both sides to the same searching player, at the budget the designer's watch uses --
  // `WATCH_AGENT`, so the page holds one opinion about who is cheap enough to watch.
  const watch = document.getElementById('watch');
  if (watch) watch.onclick = () => restart([WATCH_AGENT, WATCH_AGENT], false);
  document.getElementById('flip').onclick = () => { FLIPPED = !FLIPPED; drawBoard(); };
  // A refused take-back is answered under the button that asked for it. A rated game
  // draws no take-back at all -- the store refuses one, and a control whose only answer
  // is a refusal is worse than no control.
  const undo = document.getElementById('undo');
  if (undo) undo.onclick = () =>
    api('/api/session/' + snapshot.id + '/undo', {})
      .then(s => applySnapshot(s, false)).catch(error => fail(error, 'undo'));
  // Resigning is a finish: the row records it as a loss and, in a rated game, the rating
  // moves on it exactly as it moves on a checkmate. So it takes two presses, the first of which says
  // what the second does -- and the button disarms itself on a timer and when it loses
  // focus, because a control left armed behind the reader's back is one stray click away
  // from ending their game.
  const resign = document.getElementById('resign');
  if (resign) {
    const disarm = () => { resign.dataset.armed = ''; sayOn(resign, 'resign'); };
    resign.onblur = disarm;
    resign.onclick = () => {
      if (resign.dataset.armed !== 'yes') {
        resign.dataset.armed = 'yes';
        sayOn(resign, 'really resign?');
        setTimeout(disarm, 6000);
        return;
      }
      disarm();
      api('/api/session/' + snapshot.id + '/resign', {})
        .then(s => applySnapshot(s, false)).catch(error => fail(error, 'resign'));
    };
  }
  // Scoped to whatever this board was drawn into, because an inline one has no tabs of
  // its own and the view around it may.
  root.querySelectorAll('[data-tab]').forEach(node => {
    node.onclick = () => {
      TAB = node.dataset.tab;
      root.querySelectorAll('[data-tab]').forEach(n => n.classList.toggle('on', n === node));
      drawPane();
    };
  });

  // A game you have not seen before starts running and starts unreviewed. Coming back
  // to one you were already watching -- the browser's Back button out of the rules and
  // in again -- keeps its timeline and keeps it paused if that is how you left it,
  // because otherwise looking anything up costs you your place.
  if (TIMELINE_FOR !== snapshot.id) PAUSED = false;
  CURSOR = null;
  AT = null;
  record(snapshot);
  wireReviewKeys();
  wireFit();

  // The rules are wanted *while playing*, so they are fetched now and not when a tab is
  // first opened: a panel that takes a round trip to appear is a panel nobody opens
  // twice. It is one request per ruleset per page load either way.
  detailFor(rules.hash).then(drawPane).catch(() => {});
  drawBoard();
  drawPane();
  watchBoardSight(root);
  maybeAgent();
  wireCorr();
  pollCorr();

  // Who is playing, on a board nobody sits at. A watch raises one question -- why are
  // the moves like that -- and the seat is the answer, so the sentence belongs to every
  // watch rather than to the designer's alone (Q265). It is the server's, off the
  // picker's own labels, and absent from every game a person is in.
  noteWatching(snapshot);

  // A demonstration opens on the position before the moves it shows, once.
  const demo = snapshot.demo;
  if (demo && !demo.opened) {
    demo.opened = true;
    const beat = TIMELINE.map(entry => entry.plies).lastIndexOf(demo.start);
    if (beat >= 0 && beat < TIMELINE.length - 1) showBeat(beat);
  }
}

// The play frame (plan-visual.md §3.3), one markup for both chromes: the game's state
// above the board, the board, and this game's own controls directly under it, so all
// three are in the first screen at a desk and on a phone (the first-visit study's B1,
// F1, F2). Every slot carries `data-frame`, which is what the look check reads a page by
// and what the play frame's restyle (V4) hangs its look on.
//
// Above the board, the status strip -- whose move, *in check*, and the result once the
// game is over -- and the legend line naming every mark the board carries, the
// promotion zone's toggle at its end. Under it, the hand (pieces held, and the
// promotion chooser), then take back, flip, the view's shift where the board wraps, and
// resign last. Starting another game is not a control of this one, so new game, the
// other side and watch sit further down.
//
// `mode` is the store's word for the game (`snapshot.mode`). A sandbox game can be given
// up too (the first-visit study's F12): a lost game had take back and new game and no
// way to close it with a result. Its resignation is the same store call as a rated
// one's, and `site/rating.py::finish` leaves a sandbox row unrated, so it ends the game
// and moves nothing. The local application has no modes and no store to record a
// resignation in, so it offers none; a game nobody sits at is nobody's to give up.
//
// A game between two people (Q570, `between`) has no take-back, ever; its bar gains
// *offer a draw* and, until both have moved, *cancel the game*, with resign still last
// and still the one danger button (docs/plan-return.md §1.8).
//
// V4 gave it its look, in plan-visual.md §3.3's order: the status strip, the board, the
// control bar, the legend line. The square names stand outside the squares (`drawCoords`),
// a finished game puts its result on a banner over the board (`drawEnded`), and on a
// phone the bar is a row of icons with the word under each (`FRAME_ICONS`).
function frameHtml(inline, mode, watched, between) {
  const rated = mode === 'leaderboard';
  const ends = (rated || mode === 'sandbox') && !watched;
  return `
    <div class="row frame-status${between ? ' between' : ''}" data-frame="status"><span id="status"></span></div>
    <div class="panel board-panel"><div class="board-frame"><div class="board-box">
      <svg id="board" data-frame="board" viewBox="0 0 100 100"></svg>
      <div class="coords" id="coords" aria-hidden="true"></div></div>
      <div class="frame-ended" id="ended" hidden></div></div></div>
    <div id="hand" class="hand"></div>
    <div id="pieceSays" class="muted small piece-says" hidden></div>
    <div class="row frame-controls" data-frame="controls">
      ${inline ? `<button id="stopPlay" class="quiet" title="put the board back to editing this game"
        >${barWords('edit', 'back to editing')}</button>` : ''}
      ${rated || between ? '' : `<button id="undo" class="quiet" title="take your last move back"
        >${barWords('undo', 'take back')}</button>`}
      <button id="flip" class="quiet" title="turn the board around">${barWords('flip', 'flip')}</button>
      <span class="row shift" id="shift"></span>
      ${between && between.you !== null ? `<button id="draw" class="quiet"
        title="offer a draw; it stands until the other player accepts it or moves"
        >${barWords('draw', 'offer a draw')}</button>${between.mayCancel ? `<button id="cancel" class="quiet"
        title="call the game off; possible until both of you have moved, and it counts for nothing"
        >${barWords('cancel', 'cancel the game')}</button>` : ''}<button id="rematch" hidden
        title="challenge the other player to this game again, with the sides swapped"
        >${barWords('again', 'rematch')}</button>` : ''}
      ${ends && !(between && between.you === null) ? `<button id="resign" class="danger"
        title="${rated ? 'give the game up; it is recorded as a loss and your rating moves'
          : 'give the game up and end it here; a sandbox game counts for nothing'}"
        >${barWords('resign', 'resign')}</button>` : ''}
    </div>
    <div class="frame-legend small" data-frame="legend" id="legendRow" hidden>
      <span id="legend"></span><span id="marks"></span></div>`;
}

// The control bar's icons, drawn here rather than fetched: one 24-unit grid, a 1.8-unit
// round stroke in the button's own colour, no fill, so every icon is the same weight and
// takes the button's kind (quiet green, danger red) with it. Shown only on a phone, above
// the word, which stays (plan-visual.md §3.3: icons with a word under each, lower case).
const FRAME_ICONS = {
  undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
  flip: '<path d="M7 4v16"/><path d="m3 8 4-4 4 4"/><path d="M17 20V4"/><path d="m21 16-4 4-4-4"/>',
  resign: '<path d="M5 21V4"/><path d="M5 4h11l-2 4 2 4H5"/>',
  edit: '<path d="M15 5l4 4L9 19H5v-4z"/><path d="m13 7 4 4"/>',
  draw: '<path d="M5 12h14"/><circle cx="12" cy="7" r="1.4"/><circle cx="12" cy="17" r="1.4"/>',
  cancel: '<circle cx="12" cy="12" r="8"/><path d="m9 9 6 6M15 9l-6 6"/>',
  again: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/>',
};

function frameIcon(name) {
  return `<svg class="ico" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"
    fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
    stroke-linejoin="round">${FRAME_ICONS[name]}</svg>`;
}

// A bar button's face: its icon, then its word in a span of its own, so a button that
// changes what it says (resign asking to be pressed again) changes the word and keeps
// the icon.
function barWords(icon, words) {
  return `${frameIcon(icon)}<span class="w">${esc(words)}</span>`;
}

function sayOn(button, words) {
  const word = button.querySelector ? button.querySelector('.w') : null;
  if (word) word.textContent = words; else button.textContent = words;
}

// The board's side, fitted to the screen (S2): never wider than its column, and never so
// tall that the frame's controls under it fall off the first screen. The page's own
// board is measured from where it sits on the page, so the header and the status strip
// above it count; a board inline in the designer is wherever the editor put it, so only
// the frame's own parts count there. Never smaller than a phone's board, so a short
// landscape screen scrolls rather than shrinking the game to a stamp.
const BOARD_FLOOR = 280;
// The height of the one line a tapped piece's sentences take under the board (K3).
const PIECE_SAYS_LINE = 22;
// The status strip's width under which its longest sentences (a note, the computer being
// slow, *in check*) take three lines rather than two.
const STRIP_NARROW = 560;

function fitBoard() {
  const svg = document.getElementById('board');
  if (!svg || typeof svg.getBoundingClientRect !== 'function') return;
  const box = String(svg.getAttribute('viewBox') || '').split(/\s+/).map(Number);
  const panel = svg.closest ? svg.closest('.board-panel') : null;
  const controls = document.querySelector('[data-frame="controls"]');
  const hand = document.getElementById('hand');
  const status = document.querySelector('[data-frame="status"]');
  const legend = document.querySelector('[data-frame="legend"]');
  if (box.length !== 4 || !(box[2] > 0) || !(box[3] > 0) || !panel || !controls) return;
  const tall = node => (node && node.getBoundingClientRect
    ? node.getBoundingClientRect().height : 0);
  // The panel's own padding and border, round the board, and the margin the square names
  // stand in.
  const rim = tall(panel) - tall(svg);
  // Under the board: the hand, the control bar and the legend line, the frame's last
  // two slots, both held to the first screen (S2).
  // Read off where they end, so their margins and a legend that wraps count too.
  const bottom = node => (node && !node.hidden && node.getBoundingClientRect
    ? node.getBoundingClientRect().bottom : 0);
  const panelBottom = panel.getBoundingClientRect().bottom;
  const last = Math.max(bottom(hand), bottom(controls), bottom(legend));
  // A line is kept for a tapped piece's sentences (K3, `drawPieceSays`), so picking a
  // piece never pushes the bar off the screen; it is counted where it is already shown.
  const says = document.getElementById('pieceSays');
  const spare = says && says.hidden ? PIECE_SAYS_LINE : 0;
  const under = (last > panelBottom ? last - panelBottom : tall(hand) + tall(controls)) + 12
    + spare;
  const above = BOARD_INLINE
    ? tall(status) + 16
    : panel.getBoundingClientRect().top + (window.scrollY || 0);
  const side = Math.max(BOARD_FLOOR, window.innerHeight - above - rim - under);
  // The width that gives the board that height, plus the panel round it.
  const wide = side * box[2] / box[3];
  panel.style.maxWidth = Math.ceil(wide + (panel.offsetWidth - svg.clientWidth || 0)) + 'px';
  // The strip above is the board's own, so it is no wider than the board: a verdict boxed
  // across the whole column read as a line of the page rather than of this game.
  if (status && status.style) status.style.maxWidth = panel.style.maxWidth;
  // Its height is fixed, so the board never moves when the status changes (Q725), and
  // sized here, once a size: a strip narrower than `STRIP_NARROW` keeps a third line,
  // which is what its longest states take on a phone.
  if (status && status.classList) {
    status.classList.toggle('narrow', tall(status) > 0
      && status.getBoundingClientRect().width < STRIP_NARROW);
  }
}

let FIT_WIRED = false;
function wireFit() {
  if (FIT_WIRED || typeof window.addEventListener !== 'function') return;
  FIT_WIRED = true;
  window.addEventListener('resize', () => { if (SESSION) fitBoard(); });
}

function applySnapshot(snapshot, animate = true) {
  const previous = SHOWN;
  keepDepths(SESSION, snapshot);
  SESSION = snapshot;
  SELECTED = null;
  DROP = null;
  NOTE = '';
  // A take-back is not a ply, so it leaves no trail: the marks that say where a wild
  // piece walked belong to a move that was made, and the position being restored is one
  // from before it.
  if (!animate) PHASE = null;
  // The game is recorded whether or not anyone is watching it happen. A reply that
  // lands while somebody is reading move six is still a move that was played, and
  // `record` reads the position it was played in off the timeline rather than off the
  // board, so a reader's place in the game cannot change what gets written down.
  record(snapshot);
  // A take-back rewrites the game being reviewed, so it ends the review.
  if (!animate && AT) { CURSOR = null; AT = null; }
  // Reviewing means the board is showing a position the game has already left, and a
  // new ply must not drag it forward. Everything that grew still grows: the move list,
  // and the review row's count of how far there now is to go.
  if (AT) { drawPane(); drawReview(); }
  else { drawBoard(animate ? previous : null); drawPane(); }
  maybeAgent();
}

// An agent that fails has to say so. This used to `.catch(fail)`, which prepends an error
// box and leaves everything else exactly as it was -- so the status still said "thinking",
// with a spinner, forever, and `onSquare` refuses human input while `awaitingAgent` is
// set. The game was unrecoverable and the screen said it was working. Reported from a
// real session against a stale build of the extension, where every agent move raised.
//
// **A reply belongs to the game it was asked of.** `new game` and `play the other side`
// both start a second session while a reply to the first is still on its way, and the
// board itself is gone the moment the page moves on. Neither used to be checked, so an
// abandoned game went on playing into whatever had replaced it: two machine games taking
// alternate plies on one board, pieces from the position twenty moves back appearing
// among the pieces of the new one, and captures with nothing on the board to explain
// them. Watched here as `first@53, second@1, first@54, second@2`, which is precisely
// what it looks like -- and the reason it never stopped is that each chain paints and
// then asks for the next move, so replacing the game starts a second chain rather than
// ending the first.
//
// **Paused means paused, and this is the only place it can be.** A machine game is that
// chain and nothing else: there is no timer and no loop to stop, so the game runs for
// exactly as long as something keeps asking for the next move. A reply already on its
// way still lands and is still written into the timeline -- it is a move that was
// played -- and stops there, because this is what would have asked for the one after it.
function maybeAgent() {
  if (PAUSED) return;
  if (!SESSION || !SESSION.awaitingAgent) return;
  // Nobody is looking: the tab is in the background or the board is scrolled out of
  // sight. The chain waits, and `boardSeen` starts it again through here (Q541: a watch
  // in the designer's bench asked 14 times in 8 s with its board off screen).
  if (boardUnseen()) return;
  const status = document.getElementById('status');
  if (!status) return;                        // the board is not on screen any more
  const id = SESSION.id;
  // One question at a time per game: the board coming back into sight while an answer
  // is on its way must not ask a second one beside it.
  if (AGENT_ASKING === id) return;
  AGENT_ASKING = id;
  AGENT_SLOW = null;
  const answered = () => {
    if (AGENT_ASKING === id) AGENT_ASKING = null;
    if (AGENT_SLOW === id) AGENT_SLOW = null;
  };
  const mine = () => !!SESSION && SESSION.id === id && !!document.getElementById('status');
  status.innerHTML = thinkingHtml() + (SESSION.inCheck ? checkPill() : '');
  // On the site, an answer slower than `SLOW_AFTER` is the box sharing its cores with
  // other games (`docs/plan-launch.md` §5.2, L11): the strip says so once, and that the
  // move made is kept. No count, no queue place, nothing that grows.
  if (ACCOUNTS) setTimeout(() => {
    if (AGENT_ASKING !== id || !mine()) return;
    AGENT_SLOW = id;
    drawStatus();
  }, SLOW_AFTER);
  api('/api/session/' + id + '/agent', {})
    .then(s => whenSettled(() => { answered(); if (mine()) applySnapshot(s); })).catch(error => {
    answered();
    if (!mine()) return;
    // The sentence is a toast and the way out of it goes into the status. Never a box
    // beside the status: that grew the strip and pushed the board down (Q725).
    fail(error);
    status.innerHTML = '';
    const again = document.createElement('button');
    again.textContent = 'try that move again';
    again.onclick = () => { clearToast(); maybeAgent(); };
    // The seat's name, never its agent string (P15).
    status.append(`${seatName(SESSION.seats[SESSION.sideToMove])} could not move. `, again);
  });
}

// Whether the board on screen can be seen. Two things hide it, and either stops the
// machine chain: the tab going into the background, and the board leaving the viewport,
// which is what a watch in the designer's bench does when the person scrolls to the
// cards. Not `PAUSED`, which is the person's own pause and what the pause button shows;
// this one lifts by itself when the board is back.
let AGENT_ASKING = null;   // the session an agent request is on its way for
let AGENT_SLOW = null;     // that session, once its answer has taken `SLOW_AFTER`
const SLOW_AFTER = 3000;   // ms; the plan's three seconds
const SLOW_SAID = 'The computer is busy with other games; your move is safe.';
let BOARD_OFFSCREEN = false;
let BOARD_SIGHT = null;    // the IntersectionObserver on the board being drawn

function boardUnseen() {
  return BOARD_OFFSCREEN || document.visibilityState === 'hidden';
}

// Watches the board `root` holds, in place of any board watched before. Wired from
// `showBoard`; the tab's visibility is one listener for the page.
function watchBoardSight(root) {
  if (BOARD_SIGHT) BOARD_SIGHT.disconnect();
  BOARD_SIGHT = null;
  BOARD_OFFSCREEN = false;
  if (!watchBoardSight.wired && typeof document.addEventListener === 'function') {
    watchBoardSight.wired = true;
    document.addEventListener('visibilitychange', boardSeen);
  }
  const board = root && root.querySelector ? root.querySelector('#board') : null;
  if (!board || typeof IntersectionObserver !== 'function') return;
  const watching = new IntersectionObserver(entries => {
    if (BOARD_SIGHT !== watching || !entries.length) return;
    BOARD_OFFSCREEN = !entries[entries.length - 1].isIntersecting;
    boardSeen();
  });
  BOARD_SIGHT = watching;
  watching.observe(board);
}

// Back in sight: on with the game, through the one place that asks.
function boardSeen() {
  if (!boardUnseen()) maybeAgent();
}

// A snapshot nobody asked to see waits for the board to finish showing the previous
// move. There is exactly one of those -- the agent's reply -- and it is the whole of this
// bug: an agent answers in 40-550ms, a move takes 630ms to travel, and a capture's strike
// does not *start* until the capturing piece is 80% of the way there. So every capture a
// person made was wiped before anything of it had been drawn, while every capture the
// agent made played out in full, because nothing redraws the board on a person's turn.
// Reported exactly as it looks from the other side of the screen: "it only works when
// black captures".
//
// The wait is on the paint and not on the request, so the agent thinks *while* the board
// finishes moving and the delay costs nothing it did not already cost.
function whenSettled(paint) {
  setTimeout(paint, Math.max(0, BUSY_UNTIL - performance.now()));
}

function pieceOf(value) {
  if (!value) return null;
  const index = value - 1;
  // By `sideStride`, a whole army's values: with a facing, `index / nTypes` is the
  // frame a piece reads its moves from and not the player who owns it.
  const stride = SESSION.ruleset.sideStride || SESSION.ruleset.nTypes;
  return {
    player: Math.floor(index / stride),
    type: index % SESSION.ruleset.nTypes,
    facing: Math.floor((index % stride) / SESSION.ruleset.nTypes),
  };
}

// The position on screen, which is the beat under the review cursor when there is one
// and the game's own position otherwise. Every part of the drawing goes through these
// two, because a board taken from one position with a territory count taken from
// another would be two answers to one question.
function position() { return AT ? AT.board : SESSION.board; }
function tint() { return (AT ? AT.colour : SESSION.colour) || []; }

function pieceAt(square) { return pieceOf(position()[square]); }

// Always a <g>, even for the letter fallback, so the animation has one element per piece
// to move and the two-node fallback does not have to be special-cased.
function drawPiece(piece, x, y, cell, square) {
  const cx = x + cell / 2, cy = y + cell / 2;
  // An icon pair is one drawing per player and has no third entry, so a wild piece
  // (owner 2, schema 9) is always the lettered disc, in the third tint.
  const pair = piece.player > 1 ? null : (SESSION.ruleset.icons || [])[piece.type];
  const icon = pair ? pair[piece.player] : null;
  const tag = square === undefined ? '' : ` data-piece="${square}"`;
  let inner;
  if (icon) {
    const size = cell * 0.88, corner = (cell - size) / 2;
    // A piece with a facing is drawn pointing where it faces, and nothing else turns.
    // `directional` names the types that have one; `facingAngles` gives the heading,
    // that table rather than `facing * 90`, because a mirrored army turns the other
    // way round on screen. The one other rotation is ownership: a single-colour set
    // (shogi) has both armies in one colour and marks the second by turning the tile.
    const heading = (SESSION.ruleset.directional || [])[piece.type]
      ? (SESSION.ruleset.facingAngles || [])[piece.player] : null;
    const angle = heading ? heading[piece.facing || 0] || 0 : (icon.rotate ? 180 : 0);
    const spin = angle ? ` transform="rotate(${angle} ${cx} ${cy})"` : '';
    inner = `<image href="${esc(icon.href)}" x="${x + corner}" y="${y + corner}" width="${size}"
             height="${size}" preserveAspectRatio="xMidYMid meet"${spin}/>`;
  } else {
    const fill = piece.player === 0 ? 'var(--p0)'
               : piece.player === 1 ? 'var(--p1)' : 'var(--p2)';
    const ink = piece.player === 0 ? 'var(--p0-ink)'
              : piece.player === 1 ? 'var(--p1-ink)' : 'var(--p2-ink)';
    inner = `<circle cx="${cx}" cy="${cy}" r="${cell * 0.36}" fill="${fill}" stroke="var(--ink)"
             stroke-width="${cell * 0.025}" opacity="0.95"/>
            <text x="${cx}" y="${cy}" fill="${ink}" font-size="${cell * 0.42}" font-weight="600"
             text-anchor="middle" dominant-baseline="central"
             >${esc(SESSION.ruleset.pieceIds[piece.type])}</text>`;
  }
  return `<g${tag} pointer-events="none">${inner}</g>`;
}

// How far the view has been shifted across the seam, in whole squares. A wrapping board
// has no boundary of its own, so this is the reader's chosen frame of reference and the
// outline follows it. Zero is the frame the squares are named in and nothing else.
let SHIFT = { files: 0, ranks: 0 };
// The frame the current drawing was made in. The animation has to place a square the
// same way the drawing did, so both read this rather than working it out twice.
let FRAME = null;

function shiftView(files, ranks) {
  SHIFT = { files: SHIFT.files + files, ranks: SHIFT.ranks + ranks };
  drawBoard();
}

function resetView() {
  SHIFT = { files: 0, ranks: 0 };
  drawBoard();
}

// Whether the promotion zone is drawn. The reader's, not the game's, so it is read out
// of the browser once and survives opening another board.
let ZONES = zonesShown();

// The toggle, at the end of the legend line under the board's controls, beside the words for the
// zone it puts away. A game with no promotion gets no button at all, which is how the
// control says what it is about.
function drawMarks() {
  const row = document.getElementById('marks');
  if (!row || !SESSION) return;
  if (!SESSION.ruleset.zones) { row.innerHTML = ''; row.dataset.built = ''; showLegendRow(); return; }
  const state = ZONES ? 'on' : 'off';
  if (row.dataset.built !== state) {
    row.innerHTML = zoneToggleHtml(ZONES);
    row.dataset.built = state;
    document.getElementById('zonetoggle').onclick = toggleZones;
  }
  showLegendRow();
}

// The legend line takes no room while it has nothing to say: no marks and no toggle.
function showLegendRow() {
  const row = document.getElementById('legendRow');
  const legend = document.getElementById('legend');
  const marks = document.getElementById('marks');
  if (!row || !legend || !marks) return;
  row.hidden = !legend.innerHTML && !marks.innerHTML;
}

// What the marks on this board mean, in one line, and only the kinds the board is
// actually carrying. A ring is not self-explanatory and there is nowhere on a board to
// ask what one is for: a player who does not know that the teal ring is the square that
// wins the game is playing a different game from the one the rules describe. It is the
// frame's fourth slot, under the control bar (plan-visual.md §3.3), and `fitBoard` sizes
// the board so it is in the first screen: below the screen, where it was, every
// first-time reader misread the zone (study F1).
//
// The kinds come from the marks the drawing was given, so a kind that stops being drawn
// -- the promotion zone, which the reader can put away -- leaves the line with it, and a
// board with no marks at all gets no line. Each entry is its `MARK_KINDS` row: the key
// drawn by the same code that draws the mark, then its name and what it means.
//
// `drawn` names the kinds drawn between squares rather than on one -- a wild piece's coming
// step -- which the marks map cannot carry.
function drawLegend(marks, drawn) {
  const row = document.getElementById('legend');
  if (!row || !SESSION) return;
  const kinds = new Set(drawn || []);
  for (const square of Object.keys(marks || {})) {
    for (const mark of marks[square]) kinds.add(mark.kind);
  }
  // A seam is drawn on the board's edges rather than on a square, so it is read off the
  // rules the board was drawn from, and then named from its row like every other kind.
  const said = legendEntries(kinds, SESSION.ruleset.wrap, zoneWords(SESSION.ruleset.zones));
  row.innerHTML = said.length ? 'on the board: ' + said.join(' ') : '';
  showLegendRow();
}

// The legend's entries for a set of mark kinds, one per `MARK_KINDS` row drawn, in the
// row's own words: the play board's line above and the game page's start position (K1)
// both say them from here, so a mark has one name wherever it is drawn. A seam is read
// off the rules' `wrap`, since it lies on the board's edges rather than on a square;
// `zoneMore` is the zone's clause where the caller has one.
function legendEntries(kinds, wrap, zoneMore) {
  const seams = seamAxes(wrap);
  if (seams.files || seams.ranks || cutSeams(wrap).barrier.length) kinds.add('seam');
  const said = [];
  for (const kind of Object.keys(MARK_KINDS)) {
    if (!kinds.has(kind)) continue;
    const look = MARK_KINDS[kind];
    const more = kind === 'zone' ? zoneMore || '' : '';
    said.push(`<span class="entry" data-legend="${kind}">${markKey(kind)}<b
      style="color:${look.ink || look.colour}">${esc(look.name)}</b> ${esc(look.label)}${esc(more)}</span>`);
  }
  return said;
}

// Whose zone is whose, now the zone has no beads: each side's band named by its ranks,
// read off the compiled zones (`zone_payload`) and named as the board labels them. Said
// only where a side's zone is whole ranks, which is the only shape "rank 8" describes;
// a zone both sides share, or one cut some other way, gets no clause at all rather than
// a wrong one.
function zoneWords(zones) {
  if (!zones || !SESSION) return '';
  const W = SESSION.ruleset.width;
  const first = zones[0] || [], second = zones[1] || [];
  if (first.length === second.length && first.every((square, i) => square === second[i])) {
    return '';
  }
  const absent = new Set(SESSION.absent || []);
  const ranksOf = squares => {
    const ranks = [...new Set(squares.map(square => Math.floor(square / W)))].sort((a, b) => a - b);
    // Whole ranks only: every square of every rank named is in the zone.
    for (const rank of ranks) {
      for (let file = 0; file < W; file++) {
        const square = rank * W + file;
        if (!absent.has(square) && !squares.includes(square)) return null;
      }
    }
    return ranks.length ? ranks : null;
  };
  const words = (squares, player) => {
    const ranks = ranksOf(squares);
    if (!ranks) return null;
    const names = ranks.map(rank => squareName(0, rank).slice(1));
    const contiguous = ranks.every((rank, i) => i === 0 || rank === ranks[i - 1] + 1);
    const span = names.length === 1 ? `rank ${names[0]}`
      : contiguous ? `ranks ${names[0]}-${names[names.length - 1]}`
        : `ranks ${names.join(', ')}`;
    return `${sideName(player)}'s on ${span}`;
  };
  const both = [words(first, 0), words(second, 1)];
  if (both.some(said => !said)) return '';
  return ` (${both.join(', ')})`;
}

function toggleZones() {
  ZONES = !ZONES;
  rememberZones(ZONES);
  drawBoard();
}

// Moving the view across the seam, one square at a time. There is no boundary to be at
// the middle of, so these say which repeat is on screen rather than where the board is;
// `reset` goes back to the frame the squares take their names from.
//
// Only the axes that wrap get buttons, and a board with edges gets no row at all. The
// keys are shift and an arrow, because the bare arrows step through the game already.
function drawShift() {
  const row = document.getElementById('shift');
  if (!row || !SESSION) return;
  const wrap = wrapAxes(SESSION.ruleset.wrap);
  if (!wrap.files && !wrap.ranks) { row.innerHTML = ''; row.dataset.built = ''; return; }
  if (row.dataset.built !== 'yes') {
    const across = wrap.files ? `
      <button data-shift="-1,0" title="one file back (shift + left arrow)">&larr;</button>
      <button data-shift="1,0" title="one file on (shift + right arrow)">&rarr;</button>` : '';
    const up = wrap.ranks ? `
      <button data-shift="0,1" title="one rank up (shift + up arrow)">&uarr;</button>
      <button data-shift="0,-1" title="one rank down (shift + down arrow)">&darr;</button>` : '';
    row.innerHTML = `<span class="muted small">slide the board</span>${across}${up}
      <button id="reframe" title="back to the normal view">reset</button>
      <span class="muted small" id="frame"></span>`;
    row.dataset.built = 'yes';
    row.querySelectorAll('[data-shift]').forEach(node => {
      const by = node.dataset.shift.split(',');
      node.onclick = () => shiftView(Number(by[0]), Number(by[1]));
    });
    document.getElementById('reframe').onclick = resetView;
  }
  // Which square the repeat on screen runs from, read off the frame rather than off
  // `SHIFT` -- eight clicks along an eight-file board is back where it started.
  const corner = FRAME ? cellSquare(0, 0, FRAME) : 0;
  const W = SESSION.ruleset.width;
  const repeat = 'showing from ' + squareName(corner % W, Math.floor(corner / W));
  const reframe = document.getElementById('reframe');
  reframe.disabled = corner === 0;
  // Also on the button, because a phone's control row leaves the sentence out.
  reframe.title = 'back to the normal view (now ' + repeat + ')';
  document.getElementById('frame').textContent = repeat;
}

function drawBoard(previous) {
  const rules = SESSION.ruleset;
  const W = rules.width, H = rules.height;
  const cell = 100 / Math.max(W, H);
  const svg = document.getElementById('board');
  if (!svg) return;

  // What the ply currently arriving is made of. Worked out here rather than inside the
  // animation because the *drawing* needs it too: the squares a wild piece walked
  // between are marked, and those marks have to be there for a reader who has turned
  // motion off and have to survive a redraw for a flip or a piece highlight.
  // Never while reviewing: a step *is* a position, and the review draw hands it over
  // whole rather than as a transition to be played.
  const beats = (previous && !AT) ? plyBeats(SESSION.board, SESSION.lastMove) : null;
  if (beats) PHASE = beats.length > 1 ? beats[1].steps : null;
  const walked = new Map();
  for (const step of (PHASE || [])) { walked.set(step.from, 'from'); walked.set(step.to, 'to'); }

  // The frame this drawing is made in. Everything below places a square through it, and
  // so does the animation, because a board and a piece flying across it that disagreed
  // about where a square is would send the piece to the square it used to be on.
  FRAME = boardView(W, H, wrapAxes(rules.wrap), SHIFT, FLIPPED, rules.lattice,
                    boardCut(SESSION.absent, rules.wrap));
  svg.setAttribute('viewBox', viewBox(FRAME, cell));

  const targets = new Set(movesFrom(SELECTED).map(m => m.to));
  const dropTargets = DROP === null ? new Set()
    : new Set(SESSION.moves.filter(m => m.drop === DROP).map(m => m.to));
  const last = AT ? AT.lastMove : SESSION.lastMove;
  const colour = tint();
  const walls = new Set(SESSION.obstacles || []);
  const lit = HIGHLIGHT === null ? null : HIGHLIGHT;

  // Squares the rules single out, a list per square: the ones a game ends on when a
  // piece reaches one, and the ones a piece promotes on. Built from the compiled rules
  // the session carries, so it belongs to the game on screen and cannot outlive it.
  // `ZONES` is the reader's own toggle, which is why it is read here on every draw.
  // The check is the position's, not the rules': the square of the royal piece the engine
  // said is attacked (`checkSquares`).
  const SQUARE_MARKS = squareMarks(Object.assign({}, rules, { checked: checkSquares() }),
                                   { zone: !ZONES });
  // Each square its own piece of wood, cut at an offset hashed from the game and the
  // square (`wood.js`), so the same game is grained the same on every draw.
  const identity = rules.hash || rules.name || `${W}x${H}`;
  const wood = woodGrain('board', identity, cell);
  const framed = svg.closest ? svg.closest('.board-frame') : svg.parentNode;
  if (framed && framed.style && typeof framed.style.setProperty === 'function') {
    const at = woodFrameAt(identity);
    framed.style.setProperty('--wood-x', at[0] + 'px');
    framed.style.setProperty('--wood-y', at[1] + 'px');
  }

  // Where the squares' names go: outside the board where its drawing ends at its edge
  // (`drawCoords`), on the naming frame's own squares where ghosts or a seam's arrows
  // already stand round it.
  const outside = labelsOutside(FRAME);

  // One square, wherever in the grid it is being drawn. `depth` is zero inside the
  // outline and says how far out a ghost copy lies otherwise.
  //
  // A ghost is the same square again, so it draws the same things -- the last move's
  // mark, a selection, the dots saying where a selected piece may go -- and a click on
  // one is a click on that square. Two things it does not take: the `data-piece` tag,
  // which the animation looks a piece up by and which has to name one node, and the
  // coordinate label, which belongs to the frame of reference the outline marks.
  function cellHtml(square, u, v, depth) {
    const corner = cellXY(u, v, cell, FRAME);
    const x = corner[0], y = corner[1];
    const file = square % W, rank = Math.floor(square / W);
    const real = depth === 0;
    const shade = cellShade(file, rank, FRAME);
    let out = cellShape(x, y, cell, FRAME, `fill="${shade}" data-square="${square}"`)
      + wood.cell(x, y, FRAME, square, shade);
    if (walls.has(square)) {
      out += cellShape(x, y, cell, FRAME, 'pointer-events="none" fill="var(--wall)"');
      out += `<path d="${wallCross(x, y, cell, FRAME)}"
               stroke="var(--ink)" stroke-width="${cell * 0.06}" opacity="0.5"
               pointer-events="none"/>`;
    }
    if (colour[square]) {
      out += cellShape(x, y, cell, FRAME, `pointer-events="none"
               fill="${colour[square] === 1 ? 'var(--claim0)' : 'var(--claim1)'}" opacity="0.3"`);
    }
    // Under the piece, because a square that ends the game is a square pieces arrive
    // on, and the piece standing there has to stay the thing you see first.
    if (SQUARE_MARKS[square]) out += squareBadge(x, y, cell, SQUARE_MARKS[square], FRAME, 'board');
    if (last && (square === last.from || square === last.to)) {
      out += mark(x, y, cell, 'var(--hl-last)', 0.34);
    }
    // Where the wild piece walked, in its own hand and not the mover's: a dashed
    // outline rather than a tint, because the palette's third owner is an amber a
    // shade off the last-move mark and "the boar did this" has to be legible as a
    // *different* kind of event, not a slightly different colour. It stays on the
    // board after the motion has finished, which is what a reader who looked away
    // for a second needs in order to see what took the piece that is now missing.
    if (walked.has(square)) out += trail(x, y, cell, walked.get(square) === 'to');
    if (square === SELECTED) out += mark(x, y, cell, 'var(--hl-sel)', 0.55);
    const piece = pieceAt(square);
    if (piece) {
      if (lit !== null && piece.type === lit) out += mark(x, y, cell, 'var(--hl-sel)', 0.45);
      out += drawPiece(piece, x, y, cell, real ? square : undefined);
    }
    if (targets.has(square) || dropTargets.has(square)) {
      out += `<circle cx="${x + cell / 2}" cy="${y + cell / 2}"
               r="${piece ? cell * 0.44 : cell * 0.13}" fill="${piece ? 'none' : 'var(--dot)'}"
               stroke="var(--dot)" stroke-width="${cell * 0.05}" opacity="0.8"
               data-square="${square}"/>`;
    }
    if (real) return outside ? out : out + coordsHtml(square, x, y, cell, FRAME);
    return `<g opacity="${ghostFade(depth).toFixed(3)}">${out}</g>`;
  }

  // The board continuing past the outline first, then the repeat the reader is in, then
  // the outline over both. A board with edges has no ghosts and no outline, so it comes
  // out of this exactly as it always did.
  let out = '';
  for (const ghost of ghostCells(FRAME)) {
    out += cellHtml(ghost.square, ghost.u, ghost.v, ghost.depth);
  }
  for (let rank = H - 1; rank >= 0; rank--) {
    for (let file = 0; file < W; file++) {
      const square = rank * W + file;
      // A hexagon's corners are not cells: no path, no hatch, nothing to click.
      if (cellAbsent(square, FRAME)) continue;
      const place = squareCell(square, FRAME);
      out += cellHtml(square, place[0], place[1], 0);
    }
  }
  // The outline, then the seam over it. The outline stands down on an axis whose seam
  // is a fact of the game, so a reader is never offered two boundaries on one axis with
  // only one of them meaning anything.
  out += outlineHtml(cell, FRAME, rules.wrap) + barrierHtml(cell, FRAME, rules.wrap);
  // Where each wild piece steps next, over everything: a dotted arrow from the edge of
  // the piece to short of the square's middle, so it never covers the piece or a
  // legal-move dot. The core's answer for the position on screen (`coming`).
  const coming = AT ? AT.coming : SESSION.coming;
  out += comingHtml(coming, cell, W, H, FRAME, rules.wrap);

  svg.innerHTML = markDefs(SQUARE_MARKS, 'board') + wood.defs() + out;
  svg.onclick = event => {
    const node = event.target.closest('[data-square]');
    if (node) clickSquare(Number(node.dataset.square));
  };

  // Redrawing replaces the board's contents outright, so whatever the last move was
  // still animating is destroyed by the next draw. `BUSY_UNTIL` is when it would have
  // finished, and `whenSettled` is what respects it.
  BUSY_UNTIL = beats
    ? performance.now() + animateChange(svg, previous, cell, W, H, FRAME, beats) : 0;
  const busy = BUSY_UNTIL ? BUSY_UNTIL - performance.now() : 0;
  // What is coming is about the position the ply *leaves*, so it shows once the ply has
  // finished arriving: held back while the pieces move, then faded in.
  const arrows = svg.querySelector('[data-coming]');
  if (arrows && busy > 0 && typeof arrows.animate === 'function') {
    arrows.animate([{ opacity: 0 }, { opacity: 1 }],
                   { duration: PACE.beat, delay: busy, fill: 'backwards' });
  }
  // What the *game* is showing, which a review draw does not change: the next ply has
  // to animate from the position before it whether or not somebody was looking at move
  // six when it arrived.
  if (!AT) SHOWN = { board: SESSION.board.slice(), colour: (SESSION.colour || []).slice() };

  drawHand();
  drawStatus();
  drawPieceSays();
  drawReview();
  drawShift();
  drawMarks();
  drawLegend(SQUARE_MARKS, coming && coming.length ? ['coming'] : []);
  drawCoords(cell, outside);
  fitBoard();
}

// The squares' names in a margin of their own beside the board (plan-visual.md §3.3), in
// the numbers' face, where no piece and no mark covers them. Each is placed where
// `edgeLabels` says its edge is, as a share of the drawing so it follows the board's size
// without a redraw, and set off outwards by the frame's wood and a little more.
function drawCoords(cell, outside) {
  const layer = document.getElementById('coords');
  if (!layer) return;
  const panel = layer.closest ? layer.closest('.board-panel') : null;
  if (panel && panel.classList) panel.classList.toggle('coords-out', !!outside);
  // A hex board's names fall in the corners its cells leave bare, on the frame's wood, so
  // each is set close to its own cell's edge, on a scrap of the page's paper.
  layer.classList.toggle('on-wood', !!(outside && FRAME.hex));
  if (!outside) { layer.innerHTML = ''; return; }
  const box = viewBox(FRAME, cell).split(/\s+/).map(Number);
  layer.innerHTML = edgeLabels(cell, FRAME).map(label => `<span class="co" style="left:${
    ((label.x - box[0]) / box[2] * 100).toFixed(3)}%;top:${
    ((label.y - box[1]) / box[3] * 100).toFixed(3)}%;--dx:${label.dx.toFixed(3)};--dy:${
    label.dy.toFixed(3)}">${esc(label.text)}</span>`).join('');
}

// --------------------------------------------------------------- the animation
//
// Driven by a diff of the two positions rather than by the move encoding, which is what
// makes it general: castling moves two pieces, en passant removes one that is not on the
// destination square, an atomic capture clears a neighbourhood, and a drop adds a piece
// that came from nowhere. None of the four is special-cased here, because a diff sees
// all of them as the same three facts -- something left a square, something arrived on
// one, something vanished.
//
// Every duration the board uses is stated here, in milliseconds, and nowhere else. It is
// deliberately unhurried -- a move is something a person has to *read*, and the pace this
// replaces was fast enough that a capture was finished before the capturing piece had
// arrived.
const PACE = {
  travelMin: 225,        // a move of no distance at all
  travelPerUnit: 21,     // ... and this much more per viewBox unit crossed
  travelMax: 630,        // ... but never longer than this, however wide the board
  impact: 0.8,           // the share of the travel after which the victim is struck
  strike: 390,           // a captured piece being knocked off the board
  burst: 330,            // the ring that marks where it was taken
  drop: 330,             // a piece arriving from a hand
  explosion: 480,        // the fiery burst that marks an explosion
  resist: 500,           // the shield shimmer on a piece that survived the blast
  rotate: 350,           // a piece turning to a new facing
  // The gap between a ply's two beats -- the move somebody made, and the wild-piece
  // phase the rules run after it. Not a delay put in to hide a race: the phase happens
  // after the move, so it is *drawn* after the move, and this is only the pause
  // between them. Shorter than the shortest travel (225ms), so a machine game does not
  // crawl; long enough that the piece that moved has landed and come to rest before
  // anything else on the board starts moving, which is the whole of what was missing.
  beat: 200,
};
const SVGNS = 'http://www.w3.org/2000/svg';
let BUSY_UNTIL = 0;
// The wild-piece phase of the ply currently on screen -- `[{from, to, eaten}]`, or
// null. Kept beside SHOWN and for the same reason: the board is redrawn for a flip or a
// piece highlight, and the marks that say where the boar came from have to survive it.
let PHASE = null;

// A ply is beats: the mover's move, and then the wild-piece phase the rules run after it
// inside the same `make`. `plyBeats` splits one, in `viz/wrapped.py`, because the offline
// replay steps through the same beats this board animates.

// Returns how long the board will be busy, in milliseconds, so that the next draw can
// wait for it rather than wiping it.
//
// The two copies a move across the seam is drawn as. The piece arriving is the one the
// board already drew, flown in from outside the far edge; the piece leaving is a copy of
// what stood on the origin square, flown out through the near one. Both are the same
// translation, one repeat apart, and they hand over in the middle -- so what a reader
// follows is one piece going off the board and coming back on, which is what the rule
// says happens, rather than a piece sliding the wrong way across the middle.
//
// `flight` is how long the arriving piece takes, so the two are one motion however far
// the board says it went.
function animateCrossing(svg, node, was, fx, fy, dx, dy, cell, arc, flight, delay) {
  const start = delay || 0;
  node.animate(crossOpacity(false), { duration: flight, delay: start, fill: 'backwards' });
  if (!was) return;
  const away = ghostAt(svg, was, fx + dx, fy + dy, cell);
  travel(away, -dx, -dy, arc, start);
  const fade = away.animate(crossOpacity(true),
                            { duration: flight, delay: start, fill: 'both' });
  fade.onfinish = () => away.remove();
}

// One beat after the other, in the order the rules ran them.
function animateChange(svg, previous, cell, W, H, view, beats) {
  if (STILL.matches) return 0;
  const before = previous.board;
  // The position the mover's own move leaves behind. For a ruleset with no wild pieces
  // that is the whole ply, and this is `SESSION.board`.
  const after = beats[0].board;
  if (!before || before.length !== after.length) return 0;
  const phase = beats.length > 1 ? beats[1].steps : null;
  // Squares the phase moved onto: the piece the *page* draws there is the wild piece,
  // so anything the mover's beat puts there needs a ghost of its own.
  const swallowed = new Set(phase ? phase.map(step => step.to) : []);
  const beat = animateMove(svg, before, after, swallowed, cell, W, H, view);
  if (!phase) return beat.busy;
  return Math.max(beat.busy, animatePhase(svg, phase, after, beat, cell, W, H, view));
}

// The first beat: the move somebody submitted, diffed against the position that move
// alone leaves behind. Unchanged in every ruleset without a wild-piece phase, where
// that position is the whole ply.
//
// Reports what the phase needs from it: `landed`, when the last piece of this beat came
// to rest, and `doomed`, the ghosts it left standing on squares the phase is about to
// take them on.
function animateMove(svg, before, after, swallowed, cell, W, H, view) {
  // A piece the mover moved onto a square the phase then took it on. The board draws
  // the wild piece there, so this beat travels a ghost instead -- and the phase strikes
  // that same ghost, which is what makes "it moved there and was eaten" one story.
  const doomed = new Map();
  // When the last piece of *this* beat comes to rest, which is when the phase may start.
  let landed = 0, busy = 0;
  const nothing = () => ({ busy, landed, doomed });

  const gone = [], came = [];
  for (let sq = 0; sq < after.length; sq++) {
    if (before[sq] === after[sq]) continue;
    if (before[sq]) gone.push(sq);
    if (after[sq]) came.push(sq);
  }
  if (!gone.length && !came.length) return nothing();

  // Rotation: a piece that changed value on the same square without moving.
  // Detect it before the travel/capture logic, animate a spin, and remove the
  // square from gone/came so the rest of the pipeline ignores it.
  const rotated = [];
  for (let i = gone.length - 1; i >= 0; i--) {
    const sq = gone[i];
    const j = came.indexOf(sq);
    if (j === -1) continue;
    const oldP = pieceOf(before[sq]), newP = pieceOf(after[sq]);
    if (!oldP || !newP || oldP.player !== newP.player || oldP.type !== newP.type) continue;
    rotated.push(sq);
    gone.splice(i, 1);
    came.splice(j, 1);
  }
  if (rotated.length && !gone.length && !came.length) {
    // Pure rotation move -- animate the spin and return.
    for (const sq of rotated) {
      const node = svg.querySelector(`[data-piece="${sq}"]`);
      if (!node) continue;
      const oldP = pieceOf(before[sq]), newP = pieceOf(after[sq]);
      const headings = (SESSION.ruleset.facingAngles || [])[newP.player] || [];
      const oldAngle = headings[oldP.facing || 0] || 0;
      let newAngle = headings[newP.facing || 0] || 0;
      // Pick the shortest arc: if old=270 and new=0, spin +90 not -270.
      let delta = newAngle - oldAngle;
      if (delta > 180) delta -= 360;
      if (delta < -180) delta += 360;
      const [bx, by] = squareXY(sq, cell, view);
      const cx = bx + cell / 2, cy = by + cell / 2;
      const img = node.querySelector('image');
      if (img) {
        img.animate([
          { transform: `rotate(${oldAngle}deg)`, transformOrigin: `${cx}px ${cy}px` },
          { transform: `rotate(${oldAngle + delta}deg)`, transformOrigin: `${cx}px ${cy}px` }
        ], { duration: PACE.rotate, easing: 'ease-in-out' });
      }
      busy = Math.max(busy, PACE.rotate);
      landed = Math.max(landed, PACE.rotate);
    }
    return nothing();
  }

  const at = sq => squareXY(sq, cell, view);
  // The step a move takes is the mover's to make, so which seams are open is asked of
  // the player whose piece arrived. It matters on a one-way board: the rook that may
  // not cross the left edge slid the long way round the board, and drawing it through
  // the seam would show a move that side is not allowed.
  const stepOf = (from, to) => wrapStep(from, to, W, H,
    wrapAllow(SESSION.ruleset.wrap, (pieceOf(after[to]) || {}).player));
  const pairs = [];
  const last = SESSION.lastMove;

  // The move the player actually made comes first, so a piece that landed where another
  // just left is matched to its own origin rather than to whatever happens to fit.
  if (last && last.from !== null && last.from !== undefined
      && gone.includes(last.from) && came.includes(last.to)) {
    pairs.push([last.from, last.to]);
  }
  // Then anything else that moved without changing what it is: the castling rook.
  for (const to of came) {
    if (pairs.some(p => p[1] === to)) continue;
    const from = gone.find(sq =>
      !pairs.some(p => p[0] === sq) && before[sq] === after[to]);
    if (from !== undefined) pairs.push([from, to]);
  }

  const moved = new Set(pairs.map(p => p[0]));
  // When each travelling piece lands, and which way it came from. Both are for the
  // capture below, which has to happen *to* the victim rather than beside it.
  const blows = new Map();
  let lastBlow = 0;
  for (const [from, to] of pairs) {
    let node = swallowed.has(to) ? null : svg.querySelector(`[data-piece="${to}"]`);
    const [fx, fy] = at(from), [tx, ty] = at(to);
    if (!node) {
      if (!swallowed.has(to)) continue;
      const piece = pieceOf(after[to]);
      if (!piece) continue;
      node = ghostAt(svg, piece, tx, ty, cell);
      doomed.set(to, node);
    }
    svg.appendChild(node);            // travelling pieces pass over stationary ones
    const step = stepOf(from, to);
    const arc = pathHops(before, from, step, W, H, view);
    // Where the piece came from, along the step rather than across the board. The two
    // agree on every square pair that does not cross a seam, which is every move of
    // every board with edges.
    const [dx, dy] = stepXY(step, cell, view);
    const flight = travel(node, -dx, -dy, arc);
    if (step.wrapped) {
      animateCrossing(svg, node, pieceOf(before[from]), fx, fy, dx, dy, cell, arc, flight);
    }
    // Struck from the direction the piece actually arrived from, which through a seam
    // is the opposite of the way the two squares sit on screen.
    blows.set(to, { at: flight * PACE.impact, dx: dx, dy: dy });
    lastBlow = Math.max(lastBlow, flight * PACE.impact);
    landed = Math.max(landed, flight);
    busy = Math.max(busy, flight);
  }

  // Whatever left a square without arriving anywhere: a capture, an en passant victim,
  // a piece caught in an explosion. It is not in the new position, so it is redrawn as a
  // ghost purely to be taken off it -- and the taking waits for whatever killed it to get
  // there. It used to start at once and take less time than the move did, so on anything
  // longer than a step the victim was gone before the capturing piece arrived and a
  // capture read as a piece quietly giving up. A victim with no attacker of its own -- an
  // en passant pawn, a bystander in an explosion -- is struck when the last travelling
  // piece lands, because that is the move that killed it.
  //
  // When the ruleset has an explosion rule, victims inside the blast radius of any
  // capture destination get the fiery explosion animation rather than the plain
  // knock-off, and immune pieces that survived the blast get a brief shield shimmer.
  const ex = SESSION.ruleset.explosion;
  const blastVictims = new Set();
  // Schema 10 (Q18): `by` and `of` narrow *which* captures explode at all, and the page
  // cannot work that out from two positions -- a capture by a piece that is not a bomb
  // looks exactly like one by a piece that is. Null means every capturer / every victim,
  // which is atomic and every ruleset written before schema 10.
  const explodes = (from, to) => {
    if (!ex) return false;
    const victim = pieceOf(before[to]);
    if (!victim) return false;                 // nothing was taken here
    if (ex.of && !ex.of[victim.type]) return false;
    if (ex.by) {
      if (from === null || from === undefined) return false;
      const capturer = pieceOf(before[from]);
      if (!capturer || !ex.by[capturer.type]) return false;
    }
    return true;
  };
  if (ex) {
    // Paired moves that landed on a square with an explosion neighbourhood.
    for (const [from, to] of pairs) {
      if (!explodes(from, to)) continue;
      for (const neighbour of (ex.blast[to] || [])) {
        if (gone.includes(neighbour) && !moved.has(neighbour)) blastVictims.add(neighbour);
      }
    }
    // Atomic capture where the capturer also dies: from→to never pairs because
    // `to` is not in `came` (nothing survives there). The capturer still has to
    // *travel* to `to` before the blast fires, so a ghost is animated along the
    // move path and then exploded at the destination. The victim at `to` and
    // every blast neighbour all get the explosion timed to the capturer's arrival.
    if (last && last.from !== null && last.from !== undefined
        && gone.includes(last.from) && gone.includes(last.to)
        && !came.includes(last.to) && explodes(last.from, last.to)) {
      // Animate the capturer travelling from `from` to `to` as a ghost.
      const capturerPiece = pieceOf(before[last.from]);
      if (capturerPiece) {
        const [fx, fy] = at(last.from), [tx, ty] = at(last.to);
        // Nothing of the capturer survives, so the mover is read off the piece that
        // left rather than off a piece that arrived.
        const step = wrapStep(last.from, last.to, W, H,
          wrapAllow(SESSION.ruleset.wrap, capturerPiece.player));
        const arc = pathHops(before, last.from, step, W, H, view);
        const [dx, dy] = stepXY(step, cell, view);
        const ghost = document.createElementNS(SVGNS, 'g');
        ghost.innerHTML = drawPiece(capturerPiece, tx, ty, cell);
        ghost.setAttribute('pointer-events', 'none');
        svg.appendChild(ghost);
        const flight = travel(ghost, -dx, -dy, arc);
        if (step.wrapped) {
          animateCrossing(svg, ghost, capturerPiece, fx, fy, dx, dy, cell, arc, flight);
        }
        const arriveAt = flight * PACE.impact;
        // The capturer explodes at `to` on arrival, not at `from`.
        const capBlow = { at: arriveAt, dx: dx, dy: dy };
        busy = Math.max(busy,
          explode(svg, capturerPiece, tx, ty, cell, capBlow));
        ghost.animate([{ opacity: 1 }, { opacity: 0 }],
          { duration: PACE.explosion, delay: arriveAt, fill: 'forwards' });
        // All other blast victims are timed to this arrival.
        lastBlow = Math.max(lastBlow, arriveAt);
        busy = Math.max(busy, flight);
      }
      blastVictims.add(last.to);    // the victim
      // `last.from` is NOT added: the capturer was animated above at `to`.
      moved.add(last.from);         // skip the ghost-at-origin loop below
      for (const neighbour of (ex.blast[last.to] || [])) {
        if (gone.includes(neighbour) && !moved.has(neighbour)) blastVictims.add(neighbour);
      }
    }
  }

  for (const sq of gone) {
    if (moved.has(sq)) continue;
    const piece = pieceOf(before[sq]);
    if (!piece) continue;
    const [x, y] = at(sq);
    if (blastVictims.has(sq)) {
      busy = Math.max(busy,
        explode(svg, piece, x, y, cell, blows.get(sq) || { at: lastBlow, dx: 0, dy: 0 }));
    } else {
      busy = Math.max(busy,
        capture(svg, piece, x, y, cell, blows.get(sq) || { at: lastBlow, dx: 0, dy: 0 }));
    }
  }

  // Immune pieces that were inside the blast radius but survived: a brief shield
  // shimmer tells the player the piece was in danger and resisted.  Only fires when
  // an explosion actually occurred -- not on quiet moves or regular captures.
  if (ex && blastVictims.size > 0) {
    // Blast centers are only destinations where a capture triggered an explosion,
    // not every paired move destination (which includes quiet moves).
    const blastCenters = new Set();
    for (const [from, to] of pairs) {
      if (gone.includes(to) && explodes(from, to)) blastCenters.add(to);
    }
    if (last && last.from !== null && last.from !== undefined
        && gone.includes(last.to) && !came.includes(last.to)
        && explodes(last.from, last.to)) {
      blastCenters.add(last.to);
    }
    // The shield shimmer fires at the same time as the explosion, so the player
    // reads "blast + survived" as one event rather than two separate beats.
    const resistDelay = lastBlow;
    const arrivedAt = new Set(pairs.map(p => p[1]));
    const shimmered = new Set();
    for (const center of blastCenters) {
      for (const neighbour of (ex.blast[center] || [])) {
        if (shimmered.has(neighbour)) continue;
        if (blastVictims.has(neighbour) || moved.has(neighbour)) continue;
        // Don't shimmer a square where a piece just arrived via travel animation.
        if (arrivedAt.has(neighbour)) continue;
        if (!after[neighbour]) continue;
        const nPiece = pieceOf(after[neighbour]);
        if (!nPiece) continue;
        if (!ex.immune[nPiece.type]) continue;
        shimmered.add(neighbour);
        const [nx, ny] = at(neighbour);
        busy = Math.max(busy, resist(svg, nx, ny, cell, resistDelay));
      }
    }
  }

  // And whatever arrived from nowhere: a drop.
  for (const sq of came) {
    if (pairs.some(p => p[1] === sq)) continue;
    if (swallowed.has(sq)) continue;
    const node = svg.querySelector(`[data-piece="${sq}"]`);
    if (!node) continue;
    node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: PACE.drop, easing: 'ease-out' });
    landed = Math.max(landed, PACE.drop);
    busy = Math.max(busy, PACE.drop);
  }
  return nothing();
}

// The second beat: the wild-piece phase, which the rules run after the mover's move
// inside the same `make`.
//
// The board runs it in that order too -- nothing here starts until the piece that moved
// has landed and stood still for `PACE.beat`. That is the fix for "a boar was captured
// by nothing": with both changes drawn as one transition there was no moment at which
// the boar was the thing moving, and the piece it ate went out beside a player's move
// that had not touched it.
//
// A wild piece is already drawn on the square it ends on, so its travel is the same
// backwards translation every other move gets, held at its origin until its beat begins,
// which is also what keeps it standing still while the mover moves in front of it.
function animatePhase(svg, phase, board, beat, cell, W, H, view) {
  const at = sq => squareXY(sq, cell, view);
  // A wild piece belongs to neither seat, so it crosses whatever the board's seams are
  // rather than one player's share of them.
  const allow = wrapAllow(SESSION.ruleset.wrap, null);
  const start = beat.landed + PACE.beat;
  let busy = 0;
  for (const step of phase) {
    const node = svg.querySelector(`[data-piece="${step.to}"]`);
    if (!node) continue;
    const [fx, fy] = at(step.from), [tx, ty] = at(step.to);
    svg.appendChild(node);
    const walk = wrapStep(step.from, step.to, W, H, allow);
    const arc = pathHops(board, step.from, walk, W, H, view);
    const [dx, dy] = stepXY(walk, cell, view);
    const flight = travel(node, -dx, -dy, arc, start);
    if (walk.wrapped) {
      animateCrossing(svg, node, pieceOf(board[step.to]), fx, fy, dx, dy, cell, arc,
                      flight, start);
    }
    busy = Math.max(busy, start + flight);
    if (!step.eaten) continue;
    const piece = pieceOf(step.eaten);
    if (!piece) continue;
    // Struck by the wild piece, from the direction it came: the same blow a player's
    // capture gets, so a viewer reads the two as the same kind of event.
    const blow = { at: start + flight * PACE.impact, dx: dx, dy: dy };
    const ghost = beat.doomed.get(step.to) || ghostAt(svg, piece, tx, ty, cell);
    busy = Math.max(busy, strike(svg, ghost, tx, ty, cell, blow));
  }
  return busy;
}

// Returns how long the move takes, which is what a capture on the destination square
// has to wait for.
//
// `delay` is which beat of the ply this is. It is `backwards`-filled on purpose: a piece
// whose beat has not started yet is drawn where it *was*, not where it ends up, which is
// what lets a wild piece stand still on the square it came from while the player's move
// plays out in front of it.
function travel(node, dx, dy, arc, delay = 0) {
  const distance = Math.hypot(dx, dy);
  const duration = Math.min(PACE.travelMax, PACE.travelMin + distance * PACE.travelPerUnit);
  const frames = [{ transform: `translate(${dx}px, ${dy}px)` }];
  if (arc) {
    // A lift and a slight swell, so a leap reads as going over rather than through.
    const lift = Math.min(distance * 0.45, 6);
    frames.push({ transform: `translate(${dx / 2}px, ${dy / 2 - lift}px) scale(1.12)`, offset: 0.5 });
  }
  frames.push({ transform: 'translate(0px, 0px)' });
  node.animate(frames, { duration, delay, fill: delay ? 'backwards' : 'none',
                         easing: arc ? 'ease-in-out' : 'cubic-bezier(.3,.7,.3,1)' });
  return duration;
}

// A capture, in two parts, because either one alone is easy to miss: the victim is
// knocked off the square it was standing on, and a ring is struck where it stood. The
// ring is the part that reads at a glance on a crowded board, and it is the reason this
// is not simply a fade -- a piece disappearing is what *any* of these three facts looks
// like, and a capture is the one worth telling apart.
//
// `blow` says when the piece that killed this one arrives and which way it came from, so
// both halves are timed to the impact rather than to the redraw. Returns when the last of
// it finishes.
function capture(svg, piece, x, y, cell, blow) {
  return strike(svg, ghostAt(svg, piece, x, y, cell), x, y, cell, blow);
}

// A piece drawn on the board that the *position* no longer has: the victim of a capture,
// or a piece that moved onto a square something else has since taken it on. It stands
// there, at full opacity, until whatever happens to it happens -- an animation that has
// not started leaves its element showing its own style.
function ghostAt(svg, piece, x, y, cell) {
  const ghost = document.createElementNS(SVGNS, 'g');
  ghost.innerHTML = drawPiece(piece, x, y, cell);
  ghost.setAttribute('pointer-events', 'none');
  ghost.style.transformOrigin = `${x + cell / 2}px ${y + cell / 2}px`;
  svg.appendChild(ghost);
  return ghost;
}

// The taking itself, done to a ghost that is already on the board. Split out from
// `capture` because a piece can travel *first* and be taken where it landed -- the phase
// eating what the mover just moved into it -- and that has to be one piece the eye can
// follow rather than one that vanishes and another that appears.
function strike(svg, ghost, x, y, cell, blow) {
  const cx = x + cell / 2, cy = y + cell / 2;
  const reach = Math.hypot(blow.dx, blow.dy) || 1;
  const kx = blow.dx / reach * cell * 0.24, ky = blow.dy / reach * cell * 0.24;

  const fade = ghost.animate([
    { opacity: 0.95, transform: 'translate(0px, 0px) scale(1)' },
    { opacity: 0.8, transform: `translate(${kx * 0.55}px, ${ky * 0.55}px) scale(1.08)`,
      offset: 0.3 },
    { opacity: 0, transform: `translate(${kx}px, ${ky}px) scale(0.5)` },
  ], { duration: PACE.strike, delay: blow.at, easing: 'ease-in', fill: 'forwards' });
  fade.onfinish = () => ghost.remove();

  const ring = document.createElementNS(SVGNS, 'circle');
  ring.setAttribute('cx', cx);
  ring.setAttribute('cy', cy);
  ring.setAttribute('r', cell * 0.4);
  ring.setAttribute('fill', 'none');
  ring.setAttribute('stroke', 'var(--bad)');
  ring.setAttribute('stroke-width', cell * 0.07);
  ring.setAttribute('pointer-events', 'none');
  // Invisible until the blow lands: an animation that has not started yet leaves the
  // element showing its own style, and a ring sitting on the victim beforehand would
  // announce the capture before it happened.
  ring.style.opacity = '0';
  ring.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ring);
  const flash = ring.animate([
    { opacity: 0.9, transform: 'scale(0.45)' },
    { opacity: 0, transform: 'scale(1.3)' },
  ], { duration: PACE.burst, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  flash.onfinish = () => ring.remove();

  return blow.at + Math.max(PACE.strike, PACE.burst);
}

// An explosion: visually distinct from a plain capture so the player reads "blast
// damage" rather than "someone took that piece". The victim is consumed in place --
// no directional knock, because nothing struck it from any particular direction --
// and the marker is a double ring in amber/orange rather than the single red one a
// capture uses, with an added filled flash at the centre for the fireball.
function explode(svg, piece, x, y, cell, blow) {
  const cx = x + cell / 2, cy = y + cell / 2;

  // The ghost piece burns away in place: a bright flash, then rapid shrink to nothing.
  const ghost = document.createElementNS(SVGNS, 'g');
  ghost.innerHTML = drawPiece(piece, x, y, cell);
  ghost.setAttribute('pointer-events', 'none');
  ghost.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ghost);
  const burn = ghost.animate([
    { opacity: 1, transform: 'scale(1)', filter: 'brightness(1)' },
    { opacity: 0.9, transform: 'scale(1.15)', filter: 'brightness(1.8)', offset: 0.25 },
    { opacity: 0, transform: 'scale(0.3)', filter: 'brightness(2)' },
  ], { duration: PACE.explosion, delay: blow.at, easing: 'ease-in', fill: 'forwards' });
  burn.onfinish = () => ghost.remove();

  // Inner fireball: a filled circle that flares bright and shrinks.
  const fireball = document.createElementNS(SVGNS, 'circle');
  fireball.setAttribute('cx', cx);
  fireball.setAttribute('cy', cy);
  fireball.setAttribute('r', cell * 0.32);
  fireball.setAttribute('fill', 'var(--fx-fire)');
  fireball.setAttribute('pointer-events', 'none');
  fireball.style.opacity = '0';
  fireball.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(fireball);
  const flare = fireball.animate([
    { opacity: 0.85, transform: 'scale(0.5)' },
    { opacity: 0.5, transform: 'scale(1.1)', offset: 0.4 },
    { opacity: 0, transform: 'scale(1.4)' },
  ], { duration: PACE.explosion, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  flare.onfinish = () => fireball.remove();

  // Outer blast ring: expands wide and fades, amber-coloured.
  const ring1 = document.createElementNS(SVGNS, 'circle');
  ring1.setAttribute('cx', cx);
  ring1.setAttribute('cy', cy);
  ring1.setAttribute('r', cell * 0.35);
  ring1.setAttribute('fill', 'none');
  ring1.setAttribute('stroke', 'var(--fx-flare)');
  ring1.setAttribute('stroke-width', cell * 0.08);
  ring1.setAttribute('pointer-events', 'none');
  ring1.style.opacity = '0';
  ring1.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ring1);
  const blast1 = ring1.animate([
    { opacity: 0.9, transform: 'scale(0.4)' },
    { opacity: 0, transform: 'scale(1.6)' },
  ], { duration: PACE.explosion, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  blast1.onfinish = () => ring1.remove();

  // Inner ring: a second, tighter ring for the double-ring look.
  const ring2 = document.createElementNS(SVGNS, 'circle');
  ring2.setAttribute('cx', cx);
  ring2.setAttribute('cy', cy);
  ring2.setAttribute('r', cell * 0.25);
  ring2.setAttribute('fill', 'none');
  ring2.setAttribute('stroke', 'var(--fx-spark)');
  ring2.setAttribute('stroke-width', cell * 0.06);
  ring2.setAttribute('pointer-events', 'none');
  ring2.style.opacity = '0';
  ring2.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ring2);
  const blast2 = ring2.animate([
    { opacity: 0.8, transform: 'scale(0.5)' },
    { opacity: 0, transform: 'scale(1.35)' },
  ], { duration: PACE.explosion * 0.8, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  blast2.onfinish = () => ring2.remove();

  return blow.at + PACE.explosion;
}

// A resist shimmer: the piece was inside the blast radius but is immune to the
// explosion. A translucent shield briefly pulses around it, so the player sees
// "that piece was in danger and survived" rather than "nothing happened there".
function resist(svg, x, y, cell, delay) {
  const cx = x + cell / 2, cy = y + cell / 2;

  // A rounded-rect shield outline that pulses once and fades.
  const shield = document.createElementNS(SVGNS, 'rect');
  const pad = cell * 0.1;
  shield.setAttribute('x', x + pad);
  shield.setAttribute('y', y + pad);
  shield.setAttribute('width', cell - pad * 2);
  shield.setAttribute('height', cell - pad * 2);
  shield.setAttribute('rx', cell * 0.15);
  shield.setAttribute('ry', cell * 0.15);
  shield.setAttribute('fill', 'none');
  shield.setAttribute('stroke', 'var(--fx-shield)');
  shield.setAttribute('stroke-width', cell * 0.07);
  shield.setAttribute('pointer-events', 'none');
  shield.style.opacity = '0';
  shield.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(shield);
  const pulse = shield.animate([
    { opacity: 0, transform: 'scale(1.15)' },
    { opacity: 0.75, transform: 'scale(1)', offset: 0.3 },
    { opacity: 0.6, transform: 'scale(1.05)', offset: 0.6 },
    { opacity: 0, transform: 'scale(1.1)' },
  ], { duration: PACE.resist, delay, easing: 'ease-in-out', fill: 'forwards' });
  pulse.onfinish = () => shield.remove();

  // A faint filled overlay that flashes to reinforce the "shielded" read.
  const glow = document.createElementNS(SVGNS, 'rect');
  glow.setAttribute('x', x + pad);
  glow.setAttribute('y', y + pad);
  glow.setAttribute('width', cell - pad * 2);
  glow.setAttribute('height', cell - pad * 2);
  glow.setAttribute('rx', cell * 0.15);
  glow.setAttribute('ry', cell * 0.15);
  glow.setAttribute('fill', 'var(--fx-shield)');
  glow.setAttribute('pointer-events', 'none');
  glow.style.opacity = '0';
  glow.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(glow);
  const flash = glow.animate([
    { opacity: 0 },
    { opacity: 0.2, offset: 0.3 },
    { opacity: 0 },
  ], { duration: PACE.resist, delay, easing: 'ease-in-out', fill: 'forwards' });
  flash.onfinish = () => glow.remove();

  return delay + PACE.resist;
}

// ---------------------------------------------------------------- the side panel

function drawPane() {
  const pane = document.getElementById('pane');
  if (!pane || !SESSION) return;
  const game = DETAIL[SESSION.ruleset.hash];

  if (TAB === 'moves') {
    pane.className = 'tabpane moves';
    // Which beat the board is showing, while reviewing. The list is how a reader finds
    // the ply they want to be at, so it has to say which one they are at -- and it
    // follows the cursor rather than the end of the game, or stepping back ten moves
    // leaves the panel scrolled to a position nobody is looking at.
    const rows = moveRows();
    const want = AT ? (AT.kind === 'phase' ? 'phase' : 'move') : null;
    const here = AT ? rows.findIndex(r => r.plies === AT.plies && r.kind === want) : -1;
    pane.innerHTML = rows.map((r, i) => {
      const names = (r.stripe ? ['stripe'] : []).concat(r.kind === 'phase' ? ['phase'] : [],
                                                        r.kind === 'result' ? ['result'] : [],
                                                        r.took ? ['took'] : [],
                                                        i === here ? ['on'] : []);
      return `<div${names.length ? ` class="${names.join(' ')}"` : ''}>${r.html}</div>`;
    }).join('') || '<div class="muted small">No moves have been played yet.</div>';
    // A game between two people states its rules under the moves (test 7).
    if (SESSION.correspondence) pane.innerHTML += corrRulesHtml();
    const row = here >= 0 && pane.children ? pane.children[here] : null;
    if (row && row.scrollIntoView) row.scrollIntoView({ block: 'nearest' });
    else pane.scrollTop = pane.scrollHeight;
    return;
  }
  pane.className = 'tabpane';
  if (!game) { pane.innerHTML = '<div class="muted small">loading...</div>'; return; }

  if (TAB === 'rules') {
    pane.innerHTML = `<ul class="rules">${
      (game.ruleLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      <details style="margin-top:10px">
        <summary class="muted small">where the pieces start</summary>
        <ul class="rules muted small" style="margin-top:6px">${
          (game.setupLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      </details>`;
    return;
  }

  pane.innerHTML = `<div class="pieces">${(game.pieces || []).map((piece, index) =>
    piecesHtml(piece, { index, selected: HIGHLIGHT === index })).join('')}</div>
    <div class="muted small" style="margin-top:8px">Click a piece to find it on the board.</div>`;
  pane.querySelectorAll('[data-type]').forEach(node => {
    node.onclick = () => {
      const type = Number(node.dataset.type);
      HIGHLIGHT = HIGHLIGHT === type ? null : type;
      drawBoard();
      drawPane();
    };
  });
}

// The move list, one row per beat of the game. A ply is the move somebody made, and in a
// game with wild pieces also the phase the rules ran after it -- which the board animates
// and the review steps onto, so a list that said nothing about it was the one place a
// wolf's capture went unrecorded. The phase's row sits under its move and reads on from
// it: *1... Pe7-e6*, then *and the wolf on c6 steps to c5*.
//
// Everything in a row is what the core reported: the move's own text, and the phase's
// steps as `lastMove.phase` carried them, with the walkers `plyBeats` found when the
// timeline split the ply. A ply the timeline never split -- nothing on this page makes
// one today -- still gets its row, calling the walker a wild piece rather than guessing
// which. A ply with no phase is one row, exactly as it always was.
//
// The stripe is per ply, not per row, so a move and its phase read as one band. A phase
// row is quieter than a move's, since most of them are a patrol going round -- except one
// in which a wild piece took something, which is the line a reader is looking for.
function moveRows() {
  const split = new Map();
  if (TIMELINE_FOR === SESSION.id) {
    for (const entry of TIMELINE) if (entry.kind === 'phase') split.set(entry.plies, entry);
  }
  const rows = [];
  const outcome = SESSION.outcome;
  const last = SESSION.history.length - 1;
  SESSION.history.forEach((m, i) => {
    const stripe = i % 2 === 0;
    rows.push({ plies: i + 1, kind: 'move', stripe: stripe, html: `${
      stripe ? `${Math.floor(i / 2) + 1}.` : '&nbsp;&nbsp;&hellip;'} ${esc(m.text)}${
      signHtml(m, i === last ? outcome : null)}${lookedHtml(m)}${corrMovedDayHtml(i)}` });
    if (!(m.phase || []).length) return;
    const beat = split.get(i + 1);
    rows.push({ plies: i + 1, kind: 'phase', stripe: stripe,
                took: m.phase.some(step => step.eaten),
                html: esc(phaseText(m.phase, beat ? beat.walkers : null)) });
  });
  // The game's end, in the strip's own words, as the list's last line.
  if (outcome) rows.push({ plies: last + 1, kind: 'result', stripe: false,
                           html: `<b>${esc(verdictText(outcome))}</b>` });
  return rows;
}

// The sign after a move that left a royal piece attacked: the session's `check` for that
// move, which is the core's answer (`web/play.py`). `#` where that move ended
// the game by checkmate, as the engine reported the outcome; `+` otherwise. A
// game with no royal piece has no check, and its moves carry no sign.
function signHtml(move, outcome) {
  if (!move || !move.check) return '';
  const mate = !!(outcome && outcome.reason === 'checkmate');
  return `<span class="sign" title="${mate ? 'checkmate: the game ended here'
    : 'check: this move attacked a royal piece'}">${mate ? '#' : '+'}</span>`;
}

// A phase in words, with the spec's names for the pieces (`phaseWords`, shared with the
// offline replay).
function phaseText(steps, walkers) {
  const rules = SESSION.ruleset;
  const nameOf = value => {
    const piece = pieceOf(value);
    return piece ? (rules.pieceNames || [])[piece.type] || (rules.pieceIds || [])[piece.type]
                 : null;
  };
  return phaseWords(steps, walkers, nameOf, rules.width);
}

// What to call a side, and who is sitting on it. A seat is an agent string --
// "minimax:nodes=1600" -- which is the identity the ladder rates and the record stores,
// and not what a person calls the thing beating them. The roster's `name` is what a
// sentence says (`shortName`); what the budget does belongs in the picker and under the
// seat, not here.
function sideName(player) {
  return (SESSION.ruleset.sides || [])[player] || `player ${player}`;
}

function seatName(agent) {
  return shortName(agent);
}

// Who sits in a seat, in words: the member's handle in a game between two people
// (Q570), and the seat's player name everywhere else.
function seatLabel(player) {
  const between = SESSION && SESSION.correspondence;
  if (between) {
    const handle = between.players[player];
    return between.you === player ? `${handle} (you)` : handle;
  }
  return seatName(SESSION.seats[player]);
}

// Under the name, how hard that seat thinks: the server's plain words off the budget it
// plays at (`web/play.py::strength`). A name alone left two machines reading "hard vs
// hard" with nothing to say what hard meant (Q288). Nothing under a person.
function seatHtml(player) {
  const strength = (SESSION.strength || [])[player] || '';
  // A person's seat carries their mark (R2) beside their handle.
  const between = SESSION.correspondence;
  const mark = between && between.marks ? markHtml(between.marks[player], 18) : '';
  // And links to their page (R1): only a member with a mark has one, never somebody who
  // left.
  const member = between && between.marks && between.marks[player];
  const label = member && between.players && between.players[player]
    ? `<a href="/players/${esc(between.players[player])}">${mark}${esc(seatLabel(player))}</a>`
    : `${mark}${esc(seatLabel(player))}`;
  return `<b>${esc(sideName(player))}</b>
    <span class="muted">${label}</span>${
    strength ? `<div class="muted small">${esc(strength)}</div>` : ''}`;
}

// How far the machine looked before it played a move, beside the move, as the core
// reported it with the move (`web/play.py::_reply`). Counted in turns, each player's move
// being one (§3.2's word for a ply), which is the unit the picker's greedy player is
// described in too; the positions a searching player looks at are what bought the turns,
// and the title says so, so the picker's number and this one are read as one thing (F7,
// P14). Nothing for a person's move, a player that does not search ahead, or a move that
// was the only one.
function lookedHtml(move) {
  const depth = move && move.depth;
  if (depth === null || depth === undefined) return '';
  const turns = `${depth} turn${depth === 1 ? '' : 's'}`;
  const said = depth === 0 ? 'played without looking ahead' : `looked ${turns} ahead`;
  const why = depth === 0
    ? 'it ran out of positions to look at before it had looked even one turn ahead, so it played a move it had not compared with the others'
    : `the positions it looks at before each move took it ${turns} ahead here; a turn is one player's move, its own or the reply`;
  return ` <span class="muted small" title="${esc(why)}">${esc(said)}</span>`;
}

// A move's depth arrives once, with the reply that played it. The site rebuilds a game
// from its stored moves on every request, and since Q424 it stores each depth beside its
// move (`matches.depths`, `0014_depths.sql`), so a reload says what the board said. What
// this is still for is a page left open across that deploy: a row from before it kept no
// depths, its next reply writes null for every move before that one, and the rebuilt
// game comes back without the depths this page had already been told -- so the page keeps
// them for the moves it is still showing. Nothing is worked out: a depth the page was
// never told stays unsaid.
function keepDepths(before, after) {
  if (!before || !after || before.id !== after.id) return;
  (after.history || []).forEach((move, i) => {
    const had = (before.history || [])[i];
    if (move.depth === undefined || move.depth === null) {
      if (had && had.depth !== undefined && had.depth !== null
          && had.ply === move.ply && had.text === move.text && had.by === move.by) {
        move.depth = had.depth;
      }
    }
  });
}

// What kind of game this is, under the two seats. A rated game and a sandbox game used
// to look identical, so a person could not tell which of them they were in until a
// take-back was refused -- and on the site that is the difference between a result that
// moves the ladder and one that counts for nothing.
//
// Only the store has modes. The local application sends none and gets no mark, because
// a mark that is always the same says nothing.
function modeHtml(snapshot) {
  if (snapshot.correspondence) {
    const rated = snapshot.correspondence.rated;
    return `<div style="margin-top:6px"><span class="tag" title="${rated
      ? 'a game between two people; both ratings on this game move at the end'
      : 'a game between two people on rules with no rated computer players; it is kept and counts for nothing'
    }">${rated ? 'rated' : 'practice'}</span></div>`;
  }
  if (snapshot.mode === 'leaderboard') {
    return `<div style="margin-top:6px"><span class="tag" style="border-color:var(--good);
      color:var(--good)" title="this result is rated, there is no take-back, and leaving
      resigns the game">rated</span></div>`;
  }
  if (snapshot.mode === 'sandbox') {
    return `<div style="margin-top:6px"><span class="tag"
      title="undo and restart are on and nothing here counts towards a rating"
      >sandbox</span></div>`;
  }
  return '';
}

// The engine's reasons for a game ending (`Reason::as_str` in the core), as the end of a
// sentence that starts *‹side› wins* or *Drawn*. Every reason the core has is here, and
// one it gains later reads *by the rules* until it is: a reason's key is never shown
// (P19). `resigned` is the store's word, not the engine's: a thing somebody did rather
// than a way a position ended.
const REASON_WORDS = {
  checkmate: 'by checkmate', stalemate: 'by stalemate', stalemate_loss: 'by stalemate',
  stalemate_win: 'by stalemate', royal_captured: 'by capturing the royal piece',
  royal_destroyed: 'by destroying the royal piece', annihilated: 'by taking every piece',
  repetition: 'by repetition', no_progress: 'by the no-progress rule',
  ply_limit: 'at the move limit', reached: 'by reaching the goal',
  extinction: 'by wiping out a whole kind of piece',
  bare_royal: 'by leaving the royal piece alone',
  check_limit: 'by giving check enough times', coloured: 'on ground held',
  scored: 'on points', resigned: 'by resignation', agreed: 'by agreement',
};

function reasonWords(reason) {
  return REASON_WORDS[reason] || 'by the rules';
}

// How the game ended, in one sentence: the engine's outcome, or the store's for a game
// somebody gave up. The status strip says it, directly above the board, and the move
// list ends on it.
function verdictText(outcome) {
  const parts = verdictParts(outcome);
  return parts.how ? `${parts.head} ${parts.how}.` : parts.head;
}

// The same sentence in its two halves, for the ended banner over the board, which sets
// the result in the heading face and the engine's reason under it: *who won* and *how*.
function verdictParts(outcome) {
  if (outcome.reason === 'cancelled') {
    return { head: 'Cancelled before both players had moved. It counts for nothing.', how: '' };
  }
  const how = reasonWords(outcome.reason);
  if (outcome.winner === null || outcome.winner === undefined) return { head: 'Drawn', how: how };
  const side = sideName(outcome.winner);
  return { head: `${side.charAt(0).toUpperCase()}${side.slice(1)} (${
    seatLabel(outcome.winner)}) wins`, how: how };
}

// Whose result it is, for the banner's colour: the reader's own side's win or loss, and
// neither where the reader sat on no side or on both (a watched game, two people at one
// screen, a draw). The seats the store and the session name, never worked out.
function resultFor(outcome) {
  if (outcome.winner === null || outcome.winner === undefined) return '';
  const between = SESSION.correspondence;
  const you = between ? between.you
    : SESSION.seats.filter(seat => seat === 'human').length === 1
      ? SESSION.seats.indexOf('human') : null;
  if (you === null || you === undefined) return '';
  return outcome.winner === you ? 'won' : 'lost';
}

// The ended state (plan-visual.md §3.3): a banner over the middle of the board, the board
// still to be seen round it, with the result, the engine's reason, and the two things a
// finished game offers -- the game again, by this board's own control for it and in its
// words, and a look back through it, which steps into the review the row under the board
// already offers. Only on the game as it stands: stepping back takes it away.
function drawEnded() {
  const banner = document.getElementById('ended');
  if (!banner || !SESSION) return;
  const outcome = SESSION.outcome;
  if (!outcome || AT) { banner.hidden = true; banner.innerHTML = ''; return; }
  const parts = verdictParts(outcome);
  const again = ['again', 'rematch'].map(id => document.getElementById(id))
    .find(node => node && !node.hidden);
  const key = `${outcome.winner}|${outcome.reason}|${again ? again.id : ''}|${TIMELINE.length}`;
  banner.hidden = false;
  if (banner.dataset.built === key) return;
  banner.dataset.built = key;
  const word = again ? (again.querySelector && again.querySelector('.w')
    ? again.querySelector('.w').textContent : again.textContent).trim() : '';
  banner.className = `frame-ended ${resultFor(outcome)}`;
  banner.innerHTML = `<div class="ended-head">${esc(parts.head)}</div>${
    parts.how ? `<div class="ended-how">${esc(parts.how)}</div>` : ''}<div class="row">${
    again ? `<button class="primary" data-ended="again">${esc(word)}</button>` : ''}${
    TIMELINE.length > 1 ? '<button class="quiet" data-ended="review">review</button>' : ''}</div>`;
  banner.querySelectorAll('[data-ended]').forEach(node => {
    node.onclick = () => {
      if (node.dataset.ended === 'review') stepBy(-1);
      else if (again) again.click();
    };
  });
}

// Whether the position on screen has a royal piece attacked, as the engine said it: the
// snapshot's `inCheck` for the game as it stands, and for a position the game has left,
// the `check` the session recorded with the move that made it (`web/play.py`, asked of
// the core as the move was played). A ply with a wild-piece phase is in check only once
// the phase has run, which is the position the core was asked about. A finished game is
// reported with `inCheck` false, so its last move says whether it left the loser's royal
// piece attacked -- which is what a checkmate looks like.
function checkShown() {
  if (!SESSION) return false;
  if (AT) {
    const move = AT.lastMove;
    return !!(move && move.check && !(AT.kind === 'move' && (move.phase || []).length));
  }
  if (SESSION.inCheck) return true;
  return !!(SESSION.outcome && SESSION.lastMove && SESSION.lastMove.check);
}

// Where to draw that check: the square of the side to move's royal piece, found by the
// compiled `royal` flags on the position on screen. Nothing is decided here -- the engine
// said the side is in check, and this only finds where its royal piece stands. A side
// holding several royal pieces gets no square, because which of them is attacked is the
// engine's to say and it says only that one is; the status strip says *in check* anyway.
function checkSquares() {
  if (!checkShown()) return [];
  const royal = SESSION.ruleset.royal || [];
  const side = AT ? AT.sideToMove : SESSION.sideToMove;
  const found = [];
  position().forEach((value, square) => {
    const piece = pieceOf(value);
    if (piece && piece.player === side && royal[piece.type]) found.push(square);
  });
  return found.length === 1 ? found : [];
}

// The tab a board opens on, where something other than the moves asked (K3).
let OPEN_TAB = null;

// K3 (`docs/plan-understanding.md` §4.2 step 4): a tapped piece's move sentences in one
// line under the board, the *pieces* tab's own words (`pieces[type].moves`, already sent
// with the game), so a newcomer need not switch tabs to ask what this one does.
function drawPieceSays() {
  const line = document.getElementById('pieceSays');
  if (!line || !SESSION) return;
  const game = DETAIL[SESSION.ruleset.hash];
  const piece = SELECTED !== null && !AT ? pieceAt(SELECTED) : null;
  const about = piece && game && (game.pieces || [])[piece.type];
  const moves = about ? (about.moves || []) : [];
  line.hidden = !moves.length;
  line.innerHTML = moves.length
    ? `<b>${esc(about.name || about.id)}</b>: ${moves.map(esc).join('; ')}` : '';
}

// The words *in check*, as a pill of their own in the status strip, in the alarm colour
// the board's check mark is drawn in.
function checkPill() {
  return ' <span class="pill alarm" title="a royal piece is under attack">in check</span>';
}

// A line the status strip carries until the next click or move: what a click that did
// nothing meant. Today one sentence, for a piece with no legal move.
let NOTE = '';

function thinkingHtml() {
  const mover = SESSION ? SESSION.sideToMove : 0;
  const slow = SESSION && AGENT_SLOW === SESSION.id
    ? `<span class="frame-note">${esc(SLOW_SAID)}</span>` : '';
  return `<span class="spin"></span> ${SESSION ? seatDot(mover) + esc(sideName(mover)) + ' (' +
    esc(seatName(SESSION.seats[mover])) + ') is thinking' : 'thinking'}${slow}`;
}

// The side to move as a dot in its army's colour, before its name: the strip's first
// word, read before the words are.
function seatDot(player) {
  return `<span class="seat-dot s${player === 1 ? 1 : 0}" aria-hidden="true"></span>`;
}

function drawStatus() {
  drawStrip();
  drawEnded();
}

function drawStrip() {
  const status = document.getElementById('status');
  const ground = document.getElementById('ground');
  if (!status) return;
  // The strip is the frame's first slot; the result turns the whole of it into the
  // verdict, directly above the board, where every reader was already looking.
  const strip = status.closest ? status.closest('[data-frame="status"]') : null;
  const ended = on => { if (strip && strip.classList) strip.classList.toggle('ended', on); };

  if (ground) ground.innerHTML = groundHtml();
  const undo = document.getElementById('undo');
  // A game that is over cannot be resigned, and neither can a position the game has
  // already left -- the store refuses both, and the button says so before it is pressed.
  const resign = document.getElementById('resign');
  if (resign) resign.disabled = !!AT || !!SESSION.outcome;

  // A reviewed position is one the game has already left, so neither whose move it is
  // nor how the game ended is true of what is on screen. Saying either here would be
  // describing a different board from the one being looked at; the review row says
  // where you are instead, and taking a move back is not something you can do to a
  // position you have stepped away from.
  // The strip says it is an earlier position, and whether that position had a royal
  // piece attacked; the review row under the board says which one it is.
  if (AT) {
    ended(false);
    status.innerHTML = 'an earlier position' + (checkShown() ? checkPill() : '');
    if (undo) undo.disabled = true;
    return;
  }

  if (SESSION.outcome) {
    ended(true);
    status.textContent = verdictText(SESSION.outcome);
  } else if (SESSION.correspondence) {
    ended(false);
    status.innerHTML = corrStatus();
    wireCorrHeld();
  } else {
    ended(false);
    const more = SESSION.remaining > 1
      ? ` - ${SESSION.remaining} moves left this turn` : '';
    const check = SESSION.inCheck ? checkPill() : '';
    // "thinking" is also written by `maybeAgent` as it asks. A redraw while the answer
    // is on its way (a flip, a toggle) says it again rather than blanking the strip, and
    // a paused game with a machine to move says so, because `maybeAgent` is not going
    // to be called while the game is stopped.
    status.innerHTML = SESSION.awaitingAgent
      ? (PAUSED ? 'paused' : AGENT_ASKING === SESSION.id ? thinkingHtml() : '') + check
      : `${seatDot(SESSION.sideToMove)}${esc(sideName(SESSION.sideToMove))} (${
          esc(seatLabel(SESSION.sideToMove))}) to move${check}${esc(more)}${
          NOTE ? `<span class="frame-note">${esc(NOTE)}</span>` : ''}`;
  }
  if (undo) undo.disabled = SESSION.history.length === 0;
}

// The score, where the game has one on the board itself. A territory game is won by a
// count of squares, and a player who cannot see the count is playing blind.
function groundHtml() {
  const rule = SESSION.ruleset.territory;
  const colour = tint();
  if (!rule || !colour.length) return '';
  let first = 0, second = 0;
  for (const c of colour) { if (c === 1) first++; else if (c === 2) second++; }
  const target = rule.win_at ? ` - ${rule.win_at} claims the game` : '';
  // Named, because the two numbers used to be "12 to 9" with nothing saying which was
  // whose -- and on a flipped board the reader cannot even guess from the picture.
  return `ground held: ${esc(sideName(0))} <b>${first}</b>, ${esc(sideName(1))} <b>${second}</b>
    of ${colour.length}${esc(target)}
    <span class="muted"> (${rule.mode === 'repaint'
      ? 'a claim can be taken back' : 'a claim is permanent'})</span>`;
}

function mark(x, y, cell, colour, opacity) {
  return cellShape(x, y, cell, FRAME,
    `fill="${colour}" opacity="${opacity}" pointer-events="none"`);
}

// Where a wild piece walked this ply: dashed on the square it left, solid on the one it
// stepped onto, both in the third owner's colour. Two weights rather than two colours,
// so the pair reads as a direction on a board already using amber for the last move.
function trail(x, y, cell, arrived) {
  const dash = arrived ? '' : ` stroke-dasharray="${cell * 0.13} ${cell * 0.09}"`;
  return cellShape(x, y, cell, FRAME, `fill="none" stroke="var(--p2)" stroke-width="${cell * 0.07}"
           opacity="${arrived ? 0.95 : 0.7}"${dash} pointer-events="none"`, 0.06);
}

function movesFrom(square) {
  if (square === null || !SESSION) return [];
  return SESSION.moves.filter(m => m.from === square);
}

function drawHand() {
  const hand = document.getElementById('hand');
  if (!hand) return;
  // A hand is a count per piece type, the way the state carries it -- index 2 holding 3
  // means three of piece type 2, not one piece called 3. While reviewing it is the hand
  // *that* position had: the live one belongs to a board nobody is looking at, and a
  // captured piece appearing in it twenty plies early is the same kind of lie as a piece
  // standing on the wrong square.
  const owner = AT ? AT.sideToMove : SESSION.sideToMove;
  const mine = ((AT ? AT.hands : SESSION.hands) || [])[owner] || [];
  const held = mine.map((count, type) => [type, count]).filter(([, count]) => count > 0);
  const droppable = new Set(SESSION.moves.filter(m => m.drop !== undefined).map(m => m.drop));
  if (!held.length) { hand.innerHTML = ''; return; }

  hand.innerHTML = '<span class="muted" style="align-self:center">in hand:</span>' +
    held.map(([type, count]) =>
      `<button data-drop="${type}" class="${DROP === type ? 'on' : ''}"
        ${!AT && droppable.has(type) ? '' : 'disabled'}>${esc(SESSION.ruleset.pieceIds[type])} x${count}</button>`
    ).join('');
  hand.querySelectorAll('[data-drop]').forEach(node => {
    node.onclick = () => {
      DROP = DROP === Number(node.dataset.drop) ? null : Number(node.dataset.drop);
      SELECTED = null;
      drawBoard();
    };
  });
}

function clickSquare(square) {
  // The board is showing a position the game has left; there is nothing to move in it.
  if (AT) return;
  if (!SESSION || SESSION.outcome || SESSION.awaitingAgent) return;
  if (SESSION.seats[SESSION.sideToMove] !== 'human') return;
  // Between two people, only the player to move picks, and a picked move waits to be
  // sent: the board takes no second pick until it is sent or put back.
  const between = SESSION.correspondence;
  if (between && (between.you !== SESSION.sideToMove || CORR_HELD !== null)) return;
  NOTE = '';

  if (DROP !== null) {
    const drop = SESSION.moves.find(m => m.drop === DROP && m.to === square);
    if (drop) return send(drop.move);
    DROP = null;
  }

  const candidates = movesFrom(SELECTED).filter(m => m.to === square);
  if (candidates.length === 1) return send(candidates[0].move);
  // A strike that places its survivor: one move per landing square (and per promotion).
  if (candidates.length > 1 && candidates[0].landing !== undefined)
    return chooseLanding(candidates);
  // Several moves share these two squares: a promotion, and the player picks what to
  // become. Anything else sharing a square pair would be a bug in the generator.
  if (candidates.length > 1 && candidates[0].promotion !== undefined)
    return choosePromotion(candidates);
  if (candidates.length > 1 && candidates[0].rotation !== undefined)
    return chooseRotation(candidates);

  // On the first click, if the piece has rotation moves, show the rotation chooser
  // immediately instead of requiring a second click on the same square.
  const fromHere = movesFrom(square);
  const rotations = fromHere.filter(m => m.rotation !== undefined);
  if (rotations.length > 0) {
    SELECTED = square;
    drawBoard();
    return chooseRotation(rotations);
  }

  SELECTED = fromHere.length ? square : null;
  // A click on one of your own pieces that has nowhere to go used to do nothing and say
  // nothing, and a reader in check took the board for frozen (study B1). The moves are
  // the engine's list, so *no legal move* is its answer; *in check* is its `inCheck`.
  const piece = pieceAt(square);
  NOTE = !fromHere.length && piece && piece.player === SESSION.sideToMove
    ? `that ${pieceWord(piece)} has no legal move${
        SESSION.inCheck ? ' - your side is in check' : ''}` : '';
  drawBoard();
}

function pieceWord(piece) {
  const rules = SESSION.ruleset;
  return (rules.pieceNames || [])[piece.type] || (rules.pieceIds || [])[piece.type] || 'piece';
}

function choosePromotion(candidates) {
  const hand = document.getElementById('hand');
  hand.innerHTML = '<span class="muted" style="align-self:center">promote to:</span>' +
    candidates.map(m => `<button data-move="${m.move}">${
      esc(SESSION.ruleset.pieceNames[m.promotion] || SESSION.ruleset.pieceIds[m.promotion])
    }</button>`).join('');
  hand.querySelectorAll('[data-move]').forEach(node => {
    node.onclick = () => send(Number(node.dataset.move));
  });
}

function chooseLanding(candidates) {
  const hand = document.getElementById('hand');
  hand.innerHTML = '<span class="muted" style="align-self:center">place it on:</span>' +
    candidates.map(m => `<button data-move="${m.move}">${esc(m.landingName)}${
      m.promotion !== undefined
        ? ' =' + esc(SESSION.ruleset.pieceIds[m.promotion])
        : ''
    }</button>`).join('');
  hand.querySelectorAll('[data-move]').forEach(node => {
    node.onclick = () => send(Number(node.dataset.move));
  });
}

// Arrow symbols for the four cardinal directions a piece can face.
const ROTATION_ARROWS = ['↑', '→', '↓', '←'];  // ↑ → ↓ ←
const ROTATION_LABELS = ['up', 'right', 'down', 'left'];

function chooseRotation(candidates) {
  const hand = document.getElementById('hand');
  const nf = SESSION.ruleset.nFacings || 4;
  const player = SESSION.sideToMove;
  const headings = (SESSION.ruleset.facingAngles || [])[player] || [];
  hand.innerHTML = '<span class="muted" style="align-self:center">rotate to:</span>' +
    candidates.map(m => {
      const facing = m.rotation;
      // Map the facing angle to an arrow: 0°=up, 90°=right, 180°=down, 270°=left
      const angle = headings[facing] || 0;
      const arrowIdx = Math.round(angle / 90) % 4;
      const arrow = ROTATION_ARROWS[arrowIdx];
      const label = ROTATION_LABELS[arrowIdx];
      return `<button data-move="${m.move}" title="face ${label}"
        style="font-size:1.5em;min-width:2.5em;padding:4px 8px">${arrow}</button>`;
    }).join('');
  hand.querySelectorAll('[data-move]').forEach(node => {
    node.onclick = () => send(Number(node.dataset.move));
  });
}

// A refused move is answered on the board's own control line, beside the status the
// reader is already watching, rather than at the top of a page they have scrolled past.
//
// A move whose answer never came back (Q659: the site restarting) may or may not have
// landed, and is never sent again (`api`). The page asks where the game is instead: a
// game that moved on is drawn as it is now, and one that did not is drawn as it was, with
// the selection cleared so the move can simply be made again.
function send(move) {
  SELECTED = null;
  DROP = null;
  if (SESSION.correspondence) { corrHold(move); return; }
  const id = SESSION.id, before = SESSION.history.length;
  const here = () => !!SESSION && SESSION.id === id;
  api('/api/session/' + id + '/move', { move }).then(s => applySnapshot(s))
    .catch(error => {
      // A refusal is a toast, never a box in the status strip, whose height is fixed (Q725).
      if (!error.unavailable) { fail(error); return; }
      api('/api/session/' + id).then(fresh => {
        if (!here()) return;
        if (fresh.history.length !== before) { applySnapshot(fresh); return; }
        drawBoard();
        fail(new Error(ACCOUNTS
          ? 'The site is updating or busy, so that move was not made. Try it again in a moment.'
          : 'The application did not answer, so that move was not made. Try it again.'));
      }).catch(() => {
        if (!here()) return;
        drawBoard();
        fail(error);
      });
    });
}

// ------------------------------------------------------------------ reviewing a game
//
// **A machine game is something to read, not only something to watch.** Both defects
// fixed on this surface were found by watching a game go past, which is a poor way to
// find anything: a viewer who sees something odd has no way back to it, and the game has
// moved on by the time they know they wanted to look. So the board keeps every position
// the game has been in, and a reader can stop it, walk back through them, and walk
// forward again.
//
// **The unit of a step is a beat, not a ply.** One submitted move can make two board
// changes -- the mover's, and then the wild-piece phase the rules run after it inside
// the same `make` -- and a stepper that went from ply to ply would step straight over
// the thing that was hardest to see in the first place. `plyBeats` already splits a ply
// into positions, and a beat's `board` is a real position rather than a frame of an
// animation, so the timeline is those positions, kept.
//
// **Kept, and never recomputed.** A step is an array index. Nothing here undoes a move,
// re-derives a position or works one out: every board in the list is a board the engine
// produced and the page was handed. That is what makes stepping back and then forward
// again show the same thing -- the two directions read one array in opposite orders --
// and it is also what keeps stepping on the right side of the rule this whole package
// lives inside, because reconstructing a position by undoing moves here would be the
// page answering a question that belongs to the engine.
//
// The cost is one board per beat: `width * height` small integers, so 64 numbers a beat
// on a chessboard and about 13,000 across a 200-ply game, plus the territory colouring
// and the hands where a ruleset has either. A game long enough for that to matter would
// take longer to play than a browser stays open.
let TIMELINE = [];         // every beat of this game, in order; [0] is the opening
let TIMELINE_FOR = null;   // which session it belongs to
let CURSOR = null;         // the beat on screen while reviewing; null while following
let AT = null;             // that beat itself -- what `position()` and `tint()` read
let PAUSED = false;        // the chain that plays a machine game is stopped

// One entry per beat. `plies` is how many moves had been made when this beat ended,
// which is the measure a take-back shortens the game in; `text` is the move the ply
// was, so the review line can say where you are in words rather than in an index.
// `walkers` is which piece took each of a phase's steps, as `plyBeats` found it, which
// is what lets the move list name the wolf as well as its squares.
// `coming` is where each wild piece was going next from this beat's board, the core's
// own answer (`comingAt`), so a position stepped back to shows what was coming *then*.
function beatEntry(kind, board, steps, snapshot, plies, walkers, coming) {
  const move = plies > 0 ? snapshot.history[plies - 1] : null;
  return {
    kind: kind,
    board: board.slice(),
    steps: steps || null,
    walkers: walkers || null,
    coming: coming || [],
    lastMove: move || null,
    text: move ? move.text : '',
    plies: plies,
    sideToMove: snapshot.sideToMove,
    // The colouring and the hands as the ply left them. Neither is something a diff of
    // two boards could split between a ply's beats, and no rule that can express a wild
    // piece also claims ground or takes a piece in hand -- so the two beats of a ply
    // share the ply's own, and what differs between them is the board, which is the
    // whole of what the phase changes.
    colour: (snapshot.colour || []).slice(),
    hands: (snapshot.hands || []).map(hand => hand.slice()),
  };
}

// Extend the timeline by whatever a new snapshot added, or shorten it by whatever a
// take-back removed.
//
// The position a ply started from is read off the *timeline*, not off what the board
// happens to be showing. A reply that lands while somebody is reading move six is still
// part of the game and has to be recorded from the position it was actually played in,
// and `SHOWN` at that moment is whatever the reader last looked at.
function record(snapshot) {
  const plies = snapshot.history.length;
  if (TIMELINE_FOR !== snapshot.id) { TIMELINE_FOR = snapshot.id; TIMELINE = []; }
  if (!TIMELINE.length) {
    TIMELINE = snapshot.replay
      ? replayBeats(snapshot)
      : [beatEntry('start', snapshot.board, null, snapshot, plies, null, snapshot.coming)];
    return;
  }
  // A take-back is not a move: the plies after the position being restored are no
  // longer part of the game and their beats go with them.
  while (TIMELINE.length > 1 && TIMELINE[TIMELINE.length - 1].plies > plies) TIMELINE.pop();
  const tail = TIMELINE[TIMELINE.length - 1];
  if (tail.plies >= plies) return;         // a take-back, or a redraw of what is there
  if (plies - tail.plies > 1) {
    // More than one ply between two snapshots, which nothing on this page does today.
    // A diff across several moves does not split into beats, so it is recorded as one
    // position rather than as a guess at the ones in between.
    TIMELINE.push(beatEntry('ply', snapshot.board, null, snapshot, plies, null,
                            snapshot.coming));
    return;
  }
  const beats = plyBeats(snapshot.board, snapshot.lastMove);
  beats.forEach((beat, i) => {
    TIMELINE.push(beatEntry(beat.kind, beat.board, beat.steps, snapshot, plies, beat.walkers,
                            comingAt(beats, i, snapshot.lastMove)));
  });
}

// What was coming from one beat of a ply, all of it the core's: from the board between
// the mover's move and the phase, the phase itself, which the core logged as it ran it;
// from the ply's last beat, the core's answer for the position the ply left
// (`engine.coming.coming_phase`, carried on the move as `coming`).
function comingAt(beats, i, move) {
  if (i < beats.length - 1) return beats[i + 1].steps || [];
  return (move && move.coming) || [];
}

// A session the server replayed arrives many plies in, with every position it passed
// through in `replay`. Each ply splits into beats exactly as a live one does.
function replayBeats(snapshot) {
  const at = ply => Object.assign({ history: snapshot.history }, snapshot.replay[ply]);
  const beats = [beatEntry('start', snapshot.replay[0].board, null, at(0), 0, null,
                           snapshot.replay[0].coming)];
  for (let ply = 1; ply < snapshot.replay.length; ply++) {
    const move = snapshot.history[ply - 1];
    const split = plyBeats(snapshot.replay[ply].board, move);
    split.forEach((beat, i) => {
      beats.push(beatEntry(beat.kind, beat.board, beat.steps, at(ply), ply, beat.walkers,
                           comingAt(split, i, move)));
    });
  }
  return beats;
}

// Where the cursor is: the beat on screen, or the end of the game while following it.
function cursorAt() { return CURSOR === null ? TIMELINE.length - 1 : CURSOR; }

// Step by whole beats. Out of range does nothing at all -- pressing forward at the end
// of a running game must not stop it.
function stepBy(delta) {
  const to = cursorAt() + delta;
  if (to < 0 || to > TIMELINE.length - 1) return;
  showBeat(to);
}

// Show one beat. Reading a game and playing one are different activities, so stepping
// stops the machine: a stepper that a live game kept overwriting would not be a stepper.
//
// The last beat is a special case worth stating: it *is* the position the game is in,
// so standing on it is following the game rather than reviewing it. The status line and
// the verdict are true of it, a person may move in it, and the only thing still stopped
// is the chain -- which `resumeGame` starts again.
function showBeat(index) {
  if (TIMELINE.length < 2) return;
  const at = Math.max(0, Math.min(TIMELINE.length - 1, index));
  const live = at === TIMELINE.length - 1;
  // A game somebody is playing has no pause button (B3 of `docs/archive/plan-ux.md`), so coming
  // back to the position the game is in is what starts it again. Without this, stepping
  // back in a game against the computer left it stopped with no control to restart it.
  if (live && SESSION.seats.includes('human')) return resumeGame();
  PAUSED = true;
  CURSOR = live ? null : at;
  AT = live ? null : TIMELINE[at];
  PHASE = TIMELINE[at].steps;
  SELECTED = null;
  DROP = null;
  drawBoard();
  drawPane();
}

// Back to the game as it stands, and on with it. The only way out of review, and so the
// only place the machine chain is started again -- through `maybeAgent`, which is what
// checks the reply belongs to the game on screen and that there is still a board to draw
// it on. Resuming must not start a second game beside the one already running, and it
// cannot, because it does not start a chain of its own.
function resumeGame() {
  PAUSED = false;
  CURSOR = null;
  AT = null;
  PHASE = null;
  drawBoard();
  drawPane();
  maybeAgent();
}

// Stopping does not redraw the board: a game paused a moment after a move should finish
// showing that move, and a redraw would wipe it.
function pauseGame() {
  PAUSED = true;
  drawStatus();
  drawReview();
}

// Where in the game this beat is, in words. A step is a beat, so the count is in beats;
// what a reader recognises is the move, so the move is what it leads with.
function beatLine(entry, at) {
  const where = ' - step ' + (at + 1) + ' of ' + TIMELINE.length;
  if (entry.kind === 'start') return 'the opening position' + where;
  const number = (Math.floor((entry.plies - 1) / 2) + 1) + '.' + (entry.plies % 2 ? '' : '..');
  if (entry.kind === 'phase') {
    return number + ' ' + entry.text + ', and then the wild pieces move' + where;
  }
  return 'after ' + number + ' ' + entry.text + where;
}

// The controls, and the sentence saying where you are. Words rather than glyphs alone:
// this row is the first thing a reader meets when they want to go back and look at
// something again, and "|<" is not a word.
//
// Built once and then only *updated*. Stepping redraws the board, and a board redraw
// redraws this -- so a row rebuilt from a string would replace the button under the
// pointer on every click, which drops keyboard focus after each step and makes the one
// thing this row is for, pressing back several times, the thing it is worst at.
function drawReview() {
  const row = document.getElementById('review');
  if (!row || !SESSION) return;
  // Nothing has happened yet, so there is nothing to review and no row for it.
  if (TIMELINE.length < 2) { row.innerHTML = ''; row.dataset.built = ''; return; }
  if (row.dataset.built !== 'yes') {
    row.innerHTML = `
      <button id="atStart" title="the opening position">start</button>
      <button id="stepBack" title="one step back (left arrow)">&larr; back</button>
      <button id="pause"></button>
      <button id="stepOn" title="one step on (right arrow)">forward &rarr;</button>
      <button id="atEnd" title="the position as it stands">latest</button>
      <span class="muted small" id="where"></span>`;
    row.dataset.built = 'yes';
    document.getElementById('atStart').onclick = () => showBeat(0);
    document.getElementById('stepBack').onclick = () => stepBy(-1);
    document.getElementById('stepOn').onclick = () => stepBy(1);
    document.getElementById('atEnd').onclick = () => showBeat(TIMELINE.length - 1);
    document.getElementById('pause').onclick = () => { if (PAUSED) resumeGame(); else pauseGame(); };
  }
  const at = cursorAt();
  const first = at === 0, last = at === TIMELINE.length - 1;
  document.getElementById('atStart').disabled = first;
  document.getElementById('stepBack').disabled = first;
  document.getElementById('stepOn').disabled = last;
  document.getElementById('atEnd').disabled = last;
  const pause = document.getElementById('pause');
  // Pausing is only meaningful where the game goes on by itself with nobody to stop it,
  // which is a board nobody sits at. A game you are playing already waits for you, and
  // stepping back stops it; the way on is *latest*, which is four buttons rather than a
  // fifth that reads as a second kind of turn.
  pause.hidden = !!SESSION.outcome || SESSION.seats.includes('human');
  pause.textContent = PAUSED ? 'resume' : 'pause';
  pause.title = PAUSED
    ? 'go back to the game as it stands and play on'
    : 'stop before the next move';
  document.getElementById('where').textContent = beatLine(TIMELINE[at], at);
}

// Arrow keys step, shift and an arrow move the view across the seam, space stops and
// starts. Wired once for the life of the page rather than once per game, because
// `showBoard` runs again every time a board is opened and a listener per visit is a
// listener per visit forever. It does nothing unless a board is on screen, and nothing
// while somebody is typing into a field.
//
// The shift is on the modifier because the bare arrows already step through the game,
// and reading a game is the more common of the two.
function wireReviewKeys() {
  if (wireReviewKeys.done) return;
  wireReviewKeys.done = true;
  window.addEventListener('keydown', event => {
    if (!SESSION || !document.getElementById('review')) return;
    const tag = (event.target && event.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (event.shiftKey && event.key.indexOf('Arrow') === 0) {
      const wrap = wrapAxes(SESSION.ruleset.wrap);
      const by = { ArrowLeft: [-1, 0], ArrowRight: [1, 0],
                   ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
      if (!by || (by[0] && !wrap.files) || (by[1] && !wrap.ranks)) return;
      event.preventDefault();
      shiftView(by[0], by[1]);
      return;
    }
    if (event.key === 'ArrowLeft') { event.preventDefault(); stepBy(-1); }
    else if (event.key === 'ArrowRight') { event.preventDefault(); stepBy(1); }
    else if (event.key === ' ') {
      // The key the pause button is, so it is offered exactly where that button is.
      if (SESSION.outcome || SESSION.seats.includes('human')) return;
      event.preventDefault();
      if (PAUSED) resumeGame(); else pauseGame();
    }
  });
}

// ------------------------------------------------------------- between two people
//
// Q570, `docs/plan-return.md` §1.3 and §1.5. A game between two people is a match like
// any other, with three things the page does differently, all keyed on
// `snapshot.correspondence` (`site/correspondence.py::view`):
//
// **A picked move is held until Send.** The board offers the engine's legal moves as for
// any game; picking one puts it in `CORR_HELD` and the strip says *‹move›, not sent yet* with
// *Send move* and *put it back*. Nothing reaches the server until Send, and a sent move is
// final. The board keeps the position as it stands until the move is sent, so what is on
// it is never a move the other person has not been told about.
//
// **Whose move it is, and the time left to move, are facts.** The strip says whose move
// it is, when the last move landed (*ana moved 3 hours ago*) and, in the server's words,
// how long the player to move has (*You have 4 days 6 hours to move.*, Q627: the user's
// dated exception to test 2, so a loss on time is never a surprise). The page never
// counts down: it shows the sentence it was handed, whole hours rounded down, in the
// strip's own plain text, with no ticking, no red and no emphasis that grows (§1.8's *no
// alarm red* outside resign). The rule of five days is said under the moves as a rule.
//
// **While it is the other person's move, the page asks for the match every 15 seconds,
// only while the tab is visible**, and never while it is yours (§1.5). One request at a
// time; a page that has moved on stops asking.
let CORR_HELD = null;          // the move picked and not sent yet, between two people
let CORR_POLL = null;  // the interval asking for the other person's move
const CORR_EVERY = 15000;

function stopCorrPoll() {
  if (CORR_POLL) { clearInterval(CORR_POLL); CORR_POLL = null; }
}

function corrTheirTurn() {
  const between = SESSION && SESSION.correspondence;
  return !!between && !SESSION.outcome && between.you !== null
    && between.you !== SESSION.sideToMove;
}

function pollCorr() {
  stopCorrPoll();
  if (!corrTheirTurn()) return;
  const id = SESSION.id;
  let asking = false;
  CORR_POLL = setInterval(() => {
    if (!SESSION || SESSION.id !== id || !document.getElementById('status')) {
      stopCorrPoll();
      return;
    }
    if (document.visibilityState !== 'visible' || asking || !corrTheirTurn()) return;
    asking = true;
    api('/api/session/' + id)
      .then(fresh => {
        asking = false;
        if (!fresh || !SESSION || SESSION.id !== id) return;
        if (corrChanged(SESSION, fresh)) corrApply(fresh);
      }).catch(() => { asking = false; });
  }, CORR_EVERY);
}

function corrChanged(was, now) {
  const a = was.correspondence || {};
  const b = now.correspondence || {};
  return was.history.length !== now.history.length || !!was.outcome !== !!now.outcome
    || a.drawOffer !== b.drawOffer || a.mayCancel !== b.mayCancel;
}

// A new snapshot of a game between two people: the board as `applySnapshot` draws any
// move, then the controls redrawn (an offer, a cancel that is no longer possible), and the
// polling started or stopped by whose move it now is.
function corrApply(snapshot) {
  CORR_HELD = null;
  applySnapshot(snapshot);
  refreshCorrControls();
  pollCorr();
}

function refreshCorrControls() {
  const between = SESSION.correspondence;
  const draw = document.getElementById('draw');
  if (draw) {
    const theirs = between.drawOffer !== null && between.drawOffer !== between.you;
    const mine = between.drawOffer !== null && between.drawOffer === between.you;
    sayOn(draw, theirs ? 'accept the draw' : mine ? 'draw offered' : 'offer a draw');
    draw.disabled = mine || !!SESSION.outcome;
  }
  const cancel = document.getElementById('cancel');
  if (cancel) cancel.hidden = !between.mayCancel || !!SESSION.outcome;
  // Q571. Once the game is over, *rematch* replaces the draw offer: a challenge to the
  // same person with the sides swapped, which waits for their answer like any other. A
  // game called off before it started has nothing to play again.
  const rematch = document.getElementById('rematch');
  const ended = !!SESSION.outcome && SESSION.outcome.reason !== 'cancelled';
  if (rematch) rematch.hidden = !ended;
  if (draw && SESSION.outcome) draw.hidden = true;
  drawEnded();
}

function wireCorr() {
  if (!SESSION || !SESSION.correspondence) return;
  const id = SESSION.id;
  const act = (action) => api('/api/session/' + id + '/' + action, {})
    .then(corrApply).catch(error => fail(error));
  const draw = document.getElementById('draw');
  if (draw) draw.onclick = () => {
    const between = SESSION.correspondence;
    const theirs = between.drawOffer !== null && between.drawOffer !== between.you;
    act(theirs ? 'accept' : 'draw');
  };
  // Calling the game off asks once, like resign: it ends the game for both people.
  const cancel = document.getElementById('cancel');
  if (cancel) cancel.onclick = () => {
    if (cancel.dataset.armed !== 'yes') {
      cancel.dataset.armed = 'yes';
      sayOn(cancel, 'really cancel?');
      setTimeout(() => { cancel.dataset.armed = ''; sayOn(cancel, 'cancel the game'); }, 6000);
      return;
    }
    act('cancel');
  };
  const rematch = document.getElementById('rematch');
  if (rematch) rematch.onclick = () => {
    rematch.disabled = true;
    api('/api/challenge', { rematch: id }).then(sent => {
      sayOn(rematch, 'rematch offered');
      document.getElementById('ground').textContent = `A rematch with the sides swapped `
        + `waits for ${sent.to} to accept or decline it, on your page under Challenges `
        + `you sent until then.`;
    }).catch(error => { rematch.disabled = false; fail(error, rematch); });
  };
  // Resign already goes through `/resign`; its answer redraws the controls here too.
  const resign = document.getElementById('resign');
  if (resign) {
    const pressed = resign.onclick;
    resign.onclick = () => {
      if (resign.dataset.armed === 'yes') {
        resign.dataset.armed = '';
        sayOn(resign, 'resign');
        act('resign');
        return;
      }
      pressed();
    };
  }
  refreshCorrControls();
}

function corrHold(move) {
  CORR_HELD = move;
  drawStatus();
  drawBoard();
}

function corrHeldEntry() {
  return CORR_HELD === null ? null : (SESSION.moves || []).find(m => m.move === CORR_HELD) || null;
}

function wireCorrHeld() {
  const go = document.getElementById('sendHeld');
  if (go) go.onclick = () => {
    const move = CORR_HELD;
    go.disabled = true;
    api('/api/session/' + SESSION.id + '/move', { move })
      .then(corrApply).catch(error => { CORR_HELD = null; drawStatus(); fail(error); });
  };
  const back = document.getElementById('putBack');
  if (back) back.onclick = () => { CORR_HELD = null; drawStatus(); drawBoard(); };
}

// When something happened, in words: a fact about the past, never time left.
function corrAgoWords(stamp) {
  if (!stamp) return '';
  const seconds = Math.max(0, (Date.now() - Date.parse(stamp)) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? '' : 's'} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? '' : 's'} ago`;
}

function corrStatus() {
  const between = SESSION.correspondence;
  const mover = SESSION.sideToMove;
  const check = SESSION.inCheck ? checkPill() : '';
  const last = SESSION.history.length ? SESSION.history[SESSION.history.length - 1] : null;
  const when = (between.times || [])[SESSION.history.length - 1];
  const fact = last
    ? `${esc(between.players[last.player])} moved ${esc(last.text)}${when ? ', ' + esc(corrAgoWords(when)) : ''}.`
    : `Nobody has moved yet.`;
  const offer = between.drawOffer === null ? ''
    : between.drawOffer === between.you ? ' You have offered a draw.'
    : ` ${esc(between.players[between.drawOffer])} offers a draw.`;
  const held = corrHeldEntry();
  if (held) {
    return `<b>${esc(held.text)}</b>, not sent yet. <button id="sendHeld" class="primary"
      >Send move</button> <button id="putBack">put it back</button>${check}`;
  }
  const left = between.toMove ? ` ${esc(between.toMove)}` : '';
  if (between.you === mover) {
    return `<b>Your move.</b> ${fact}${left}${offer}${check}${
      NOTE ? `<span class="frame-note">${esc(NOTE)}</span>` : ''}`;
  }
  if (between.you === null) {
    return `${esc(between.players[mover])} to move. ${fact}${left}${check}`;
  }
  return `${esc(between.players[mover])}'s move. ${fact}${left}${offer}${check}`;
}

// The day beside each move of a game between two people (s4 of the mock): when it
// landed, as a date. Nothing for any other game, which is played in one sitting.
function corrMovedDayHtml(index) {
  const between = SESSION && SESSION.correspondence;
  const when = between && (between.times || [])[index];
  if (!when) return '';
  const day = new Date(Date.parse(when));
  const said = day.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  return ` <span class="muted small" title="${esc(String(when))}">${esc(said)}</span>`;
}

function corrRulesHtml() {
  const between = SESSION.correspondence;
  return `<ul class="rules muted small" style="margin-top:10px">${
    (between.rules || []).map(rule => `<li>${esc(rule)}</li>`).join('')}</ul>`;
}
