// Page state, the router, and the helpers every view uses.

const view = document.getElementById('view');
// The header's second line as the page was served. The library, the map and what's new
// each write their own summary there; every other view shows this one, as a fresh load
// of it would, rather than whatever the last view left.
const TAGLINE_SERVED = document.getElementById('tagline').textContent;
let LIBRARY = [], OPPONENTS = [], COVERAGE = null, SESSION = null, SELECTED = null,
    DROP = null, RUN = null, TIMER = null,
    // Who you can play in the ruleset currently on screen. The ladder is the same
    // player on every board; a rule of thumb is not -- the feature vocabulary
    // normalises by board size, so one digest is not one player on two boards -- so
    // the roster has to be per-ruleset and OPPONENTS below is only the fallback for a
    // page with no game in hand.
    ROSTER = [], RATING_TIMER = null, TEST_TIMER = null, SCORING = null,
    SWEEP_TIMER = null,
    // The element `pending()` last showed, so `clearPending()` has something to remove
    // without asking the DOM to find it again.
    PENDING = null,
    // The viewport-fixed error box currently up, if any. It lives on `document.body`
    // rather than in `view`, so nothing redrawing a view takes it away and `clearToast`
    // is the only thing that does.
    TOAST = null,
    // The position the board is currently showing. Kept beside SESSION rather than
    // inside it because the animation is a diff of two snapshots, and the older one has
    // to survive the assignment that replaces it.
    SHOWN = null, TAB = 'moves', DETAIL = {}, HIGHLIGHT = null,
    // What the list can be ordered by, in words, from metrics/glossary.py. Arrives with
    // the listing, because the sort menu is the first thing a person touches.
    METRICS = null,
    // Each piece drawing the listing's thumbnails name, by stem (V6): sent once with the
    // listing rather than once a card.
    PIECE_ART = {},
    // What the store adds to a listing, or null on the local application, which has no
    // store. It carries the orderings and the filters the site's own columns support, so
    // the menu grows when the server grows a key and not before.
    SITE = null,
    // How the library is currently narrowed and ordered. Kept out of the URL on purpose:
    // it is a way of looking rather than a place, and a Back button that stepped through
    // every filter somebody tried would not be a Back button.
    // Whether the opening order has been settled against the server that answered. The
    // site opens by score and the local application by name, and which server this is
    // arrives with the first listing -- so the answer is applied once and the order-by
    // menu belongs to the reader from then on.
    ORDER_SET = false,
    BROWSE = {
      text: '', sort: 'name', descending: false, group: 'none',
      // Each axis of the kind-of-game grid, to a set of bins the reader will accept.
      // Empty means "any", which is why an untouched filter costs nothing to evaluate.
      bins: {}, stage: '', scored: '', playable: '', fairy: '', tag: '', proven: '',
      // The site's library tab (Q503b), by the key the listing sends; empty is the first.
      tab: '',
    },
    // Which way round the board is drawn. Presentation only -- a square is a square and
    // the server never hears about this -- but playing the second seat down a board with
    // your own pieces at the far end is playing it upside down, which is what made
    // taking the other side in an asymmetric game a chore rather than a choice.
    FLIPPED = false;

// Safe in text and in a quoted attribute alike, so one escaper serves every interpolation.
const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');

// An opponent's name, without the plain words a roster label carries after it: the
// server's `name` (`web/opponents.Opponent`). The long form belongs in a table; a
// sentence wants the name. The game's own roster comes first and the library-wide ladder
// is the fallback, because a card is drawn before any one game's roster has been fetched.
// A player neither names is *the computer*, never its agent string (P16); the person's
// own seat is *you*, as the ladder calls it, before any roster has arrived.
function shortName(agent) {
  const found = ROSTER.find(o => o.agent === agent)
    || OPPONENTS.find(o => o.agent === agent);
  return (found && (found.name || found.label))
    || (agent === 'human' ? 'you' : 'the computer');
}
const fmt = n => (n === null || n === undefined) ? '-' : Number(n).toFixed(3);
const STILL = window.matchMedia('(prefers-reduced-motion: reduce)');

// Every POST carries `X-Requested-With`. A cross-site form cannot set a custom header, and
// the server refuses a POST without it.
//
// A route this server withholds is refused here, in a visitor's words, so it never becomes
// a request.
//
// **Every request the page makes and reads goes through here** (Q659), because the answer
// is not always JSON. While the site is redeployed or restarted, the proxy in front of it
// answers with a web page (a 502 or 503), and `response.json()` on that throws the
// browser's own sentence: *The string did not match the expected pattern* in Safari,
// *Unexpected token <* in Chrome, *Load failed* or *Failed to fetch* for a dropped
// connection. A player read the first on every move they tried. So an answer this page
// cannot read becomes one sentence in the site's voice, the error it throws carries
// `unavailable` for a caller that can do better (a move asks where the game is), and the
// server's own sentence still wins whenever it sent one.
//
// **Only a GET is asked twice**, once, after a pause: it reads and changes nothing. A POST
// is never repeated. A move is sent as the move alone, with no position or ply beside it,
// so a repeat after a first that did land would be put to the next position, and in a
// game with drops the same move can be legal there for the other side.
let API_RETRY_PAUSE = 1500;   // ms before the one repeat of a GET

async function api(path, body) {
  const method = body === undefined ? 'GET' : 'POST';
  // Only the site withholds a route, so only a visitor ever reads this (A16).
  if (!offered(method, path)) throw new Error("That isn't available on this site.");
  const options = body === undefined
    ? {}
    : { method: 'POST', body: JSON.stringify(body),
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'chess-variant-sandbox' } };
  let answer = await askOnce(path, options);
  if (answer.unavailable && method === 'GET') {
    await new Promise(done => setTimeout(done, API_RETRY_PAUSE));
    answer = await askOnce(path, options);
  }
  if (answer.unavailable) {
    const error = new Error(unansweredWords());
    error.unavailable = true;
    throw error;
  }
  if (!answer.ok) {
    throw new Error((answer.data && answer.data.error) || answer.statusText
      || 'That did not work. Try again in a moment.');
  }
  return answer.data;
}

// One request, never thrown: what came back, or `unavailable` when nothing this page can
// read did. A refusal the server wrote (any status, with a JSON body) is an answer; an
// error page with no JSON in it is not, whatever its status.
async function askOnce(path, options) {
  let response;
  try { response = await fetch(path, options); } catch (error) { return { unavailable: true }; }
  let data;
  try { data = await response.json(); } catch (error) { return { unavailable: true }; }
  if (!response.ok && !(data && data.error) && [502, 503, 504].includes(response.status)) {
    return { unavailable: true };
  }
  return { ok: response.ok, data, statusText: response.statusText };
}

// Nothing here says the request did or did not land: a POST whose answer was lost may
// have. A caller that has found out says so itself (`send`, in `board.js`).
function unansweredWords() {
  return ACCOUNTS
    ? 'The site is updating or busy and did not answer. Try again in a moment.'
    : 'The application did not answer. Try again in a moment.';
}

// One fetch per ruleset per page load. The board page wants the rules and the pieces,
// and they do not change while a game is being played.
async function detailFor(hash) {
  if (!DETAIL[hash]) DETAIL[hash] = await api('/api/game/' + hash);
  return DETAIL[hash];
}

// The roster of opponents lives on the library endpoint, and a link straight to a game
// never asks for it -- which left the seat pickers offering nobody to play against
// except another person. Deep links are the whole point of having routes, so the roster
// cannot be a side effect of having visited the front page first.
async function roster(hash) {
  if (hash) {
    const game = await detailFor(hash);
    if (game.opponents && game.opponents.length) { ROSTER = game.opponents; return ROSTER; }
  }
  if (!OPPONENTS.length) {
    const data = await api('/api/library');
    LIBRARY = data.games; OPPONENTS = data.opponents; COVERAGE = data.coverage;
    // The same block the listing reads. Taken here too, so a deep link that never
    // passed through the library still knows whether the header has a link to offer.
    SITE = data.site || null;
    noteAdmin(SITE && SITE.admin);
  }
  return OPPONENTS;
}

// What this person has done here. Never cached: it is the one number on the page that
// changes because of something the reader just did.
async function humanFor(hash) {
  try { return await api('/api/human/' + hash); } catch (error) { return null; }
}

// The one rating table in use, and never a blend of two.
//
// A ruleset can have both: the rated ladder's, written by whatever tournament scored
// it, and the picker's own, from the on-demand run that also rates the rules of thumb.
// They are two fits over two corpora, so a column that took each agent from whichever
// table happened to have it would be inventing a comparison nobody made. The picker's
// wins when it exists because it is the one that covers everything.
function ratingTable(game) {
  if (game && game.pickerRatings && game.pickerRatings.standings) {
    return { ...game.pickerRatings, kind: 'picker' };
  }
  if (game && game.ratings && game.ratings.standings) {
    return { ...game.ratings, kind: 'ladder' };
  }
  return null;
}

function eloOf(table, agent) {
  const found = table && (table.standings || []).find(s => s.name === agent);
  return found ? found.rating : null;
}

// --------------------------------------------------------------- saying what went wrong
//
// One sentence the server sent, or one about the connection, put where the reader is
// already looking. There are exactly two places, and which one an error gets is decided
// by whether a person pressed something to cause it.
//
// **Pass the control as `near`** -- an element, or its id -- when the request was made
// from something on screen: a save, a move, a sign-in, a vote. The sentence is written
// on the control's own line, under the button rather than over it, so the answer to
// "why did that not work" is read with the thing that did not work still in sight.
//
// **Leave `near` out** when nothing was pressed: a route that would not draw, a
// background poll, a lost connection. That sentence is fixed to the viewport, because
// the page it belongs to may be scrolled anywhere. It used to be prepended to the top of
// `view`, which put every one of these above the fold of a page the reader had scrolled
// past -- reported as *"the error messages landing on top of the site, when you're
// scrolled down are easy to miss"* (the user, 2026-09-14).
//
// A later page adds neither a third place nor a box of its own; it calls this.
function fail(error, near) {
  clearPending();
  const words = errorWords(error);
  const control = typeof near === 'string' ? document.getElementById(near) : near;
  return control ? beside(control, words) : toast(words);
}

// What a failure says to the reader. A refusal is a sentence somebody wrote -- the
// server's, or the page's own `new Error('...')` -- and is shown as it is. An error the
// browser raised (a `TypeError` thrown inside a success handler, a `RangeError`, a
// `DOMException`) is a fault in this page, and its message is the engine's
// (*Cannot read properties of undefined*): the detail goes to the console for whoever
// debugs it, and the reader gets a plain sentence (Q682).
const PAGE_FAULT = 'Something went wrong on this page. Reloading it usually puts it right.';

function errorWords(error) {
  if (error instanceof Error && error.name !== 'Error') {
    try { console.error(error); } catch (ignored) { /* no console */ }
    return PAGE_FAULT;
  }
  return (error && error.message) || String(error);
}

// The controls on this page sit in a `.row` or a `.hand`, both of which wrap, so the box
// joins that line and takes a line of its own at the end of it. Dropping it next to the
// button instead would push the sentence explaining the button below the error.
//
// One answer per control: pressing a refused button twice is one refusal, twice.
function beside(control, words) {
  const anchor = control.closest('.row, .hand') || control;
  anchor.querySelectorAll('.error.beside').forEach(old => old.remove());
  const box = document.createElement('div');
  box.className = 'error beside';
  box.setAttribute('role', 'alert');
  box.textContent = words;
  anchor.append(box);
  return box;
}

// Fixed to the bottom of the viewport, and dismissed by hand rather than on a timer --
// these are the failures nobody asked for, so the reader decides when they have read it.
// One at a time, because the newest is the one still true.
function toast(words) {
  clearToast();
  const box = document.createElement('div');
  box.className = 'error toast';
  box.setAttribute('role', 'alert');
  box.textContent = words;
  const close = document.createElement('button');
  close.textContent = '\u00d7';
  close.setAttribute('aria-label', 'hide this message');
  close.onclick = clearToast;
  box.append(close);
  document.body.appendChild(box);
  TOAST = box;
  return box;
}

function clearToast() {
  if (TOAST) { TOAST.remove(); TOAST = null; }
}

// Shown the moment a click starts a transition whose one fetch is slow -- opening a
// game, going back to the library, starting a new one -- and gone the moment that view
// lands, because landing means `view.innerHTML` was reassigned and took the box with it.
// The one path that does not reassign it is failure, so `fail` clears it too.
function pending(message) {
  clearPending();
  const box = document.createElement('div');
  box.className = 'busy';
  box.innerHTML = `<span class="spin"></span> ${esc(message)}`;
  view.prepend(box);
  PENDING = box;
}

function clearPending() {
  if (PENDING) { PENDING.remove(); PENDING = null; }
}

// ------------------------------------------------------------- the way in, and the way out
//
// Signing in, a profile, the credits and the privacy notice are pages this server
// serves and nothing on this one linked to, so the way to any of them was typing the
// address. They are the site's pages rather than the local application's, which is what
// `ACCOUNTS` is for: the local application has no store and so nobody to be, and a
// header offering it a sign-in would offer a link that 404s.

let VIEWER = null,    // the handle of whoever is reading, or null for a visitor
    VIEWER_MARK = null, // their generated mark (R2), as `/api/me` sends it
    ACCOUNTS = false, // whether this server has people at all
    ADMIN = false,    // whether the reader is a maintainer, from whichever answer said so
    // Whether `/players` lists people yet (L3 of `docs/archive/plan-launch.md` §3.3): the
    // footer links it only once five people are on it. The site's fact, the
    // same for every reader.
    PLAYERS = false,
    // Whether the latest round has an entrant (L4, §3.4): the footer links the round only
    // then. Its address answers either way.
    ROUND = false;

// `/api/me` (Q199), never `/api/profile` -- that one reads this person's record on
// every rated game, which the header has no use for. `accounts` is what tells the
// local application apart from the site with nobody signed in: both answer 200 with
// the same three other fields, so a 404 is no longer how this is read.
async function identify() {
  try {
    const data = await api('/api/me');
    ACCOUNTS = data.accounts === true;
    PLAYERS = data.players === true;
    ROUND = data.round === true;
    if (data.signedIn) { VIEWER = data.handle || null; VIEWER_MARK = data.mark || null; }
    noteAdmin(data.admin);
  } catch (error) { /* offline; the header says nothing rather than something wrong */ }
}

// ------------------------------------------------------------ what this server serves
//
// The public site serves a shorter route table than the local application, and answers
// a withheld route with a 404. On the site the page draws no control for one and sends no
// request to one. `ACCOUNTS` tells the two servers apart, and the boot waits for it.
//
// The same table as `site/routes.py`'s `WITHHELD`, which `tests/test_web.py` compares.
const WITHHELD = [
  'POST /api/run', 'GET /api/run/<id>', 'GET /api/runs',
  'GET /api/scoring', 'POST /api/scoring', 'POST /api/scoring/stop',
  'POST /api/archive', 'POST /api/promote', 'POST /api/design/icon',
  'POST /api/test', 'GET /api/test/<hash>', 'POST /api/rate',
  'POST /api/design/measure', 'POST /api/design/nominate', 'GET /api/rating/<hash>',
  'POST /api/opponent', 'POST /api/opponent/forget',
  'GET /api/human', 'GET /api/human/<hash>',
];

function offered(method, path) {
  if (!ACCOUNTS) return true;
  const parts = path.split('?')[0].split('/').filter(Boolean);
  return !WITHHELD.some(route => {
    const [verb, pattern] = route.split(' ');
    const own = pattern.split('/').filter(Boolean);
    return verb === method && own.length === parts.length
      && own.every((part, i) => part.startsWith('<') || part === parts[i]);
  });
}

// Whether the reader is a maintainer, told by whichever answer arrived. The listing says
// `site.admin` and a game page says `standing.admin` -- one fact about one person, sent
// from two places because the header sits above both. Anything that is not a boolean is
// a payload that did not say, which includes every answer the local application gives,
// and leaves the flag where it was.
function noteAdmin(said) {
  if (typeof said !== 'boolean' || said === ADMIN) return;
  ADMIN = said;
  paintIdentity();
}

// The header's corner and the footer, built as elements. A handle is a name somebody
// chose for themselves, and `innerHTML` would let that name be tags.
function paintIdentity() {
  const who = document.getElementById('who'), foot = document.getElementById('foot');
  if (who) {
    who.textContent = '';
    if (ADMIN) who.append(link('/admin', 'Maintenance'), ' · ');
    // What an account adds, in one line beside *sign in* (L7, `docs/archive/plan-launch.md` §4); in
    // full on the sign-in and invitation pages (`site/pages.py` `ACCOUNT_ADDS`).
    if (ACCOUNTS && VIEWER) who.append(memberButton(VIEWER, VIEWER_MARK));
    else if (ACCOUNTS) who.append(link('/signin', 'sign in'), ' to make your games count');
  }
  // Q760: *People*, after *Games*, for a member on the site (`shell.py`'s `PEOPLE`); the
  // site's own pages draw it in the markup. A visitor would only be sent to the door, and
  // the local application has nobody.
  const nav = document.getElementById('sections');
  if (nav && ACCOUNTS && VIEWER && !nav.querySelector('a[href="/players"]')) {
    const people = link('/players', 'People');
    const games = nav.querySelector('a');
    if (games) games.after(people); else nav.append(people);
  }
  if (foot) {
    foot.textContent = '';
    // On both servers: the games and the landing page (Q718, Q723), and the changelog,
    // which is a file in the tree, so the local application has it too. None needs a
    // store behind it.
    foot.append(link(GAMES_PATH, 'games'), ' · ', link('/', 'about'), ' · ',
      link('#/whats-new', "what's new"));
    if (ACCOUNTS) {
      if (ROUND) foot.append(' · ', link('#/round', 'the round'));
      if (PLAYERS) foot.append(' · ', link('/players', 'players'));
      foot.append(' · ', link('/suggestions', 'suggestions'));
      foot.append(' · ', link('/credits', 'credits'), ' · ', link('/privacy', 'privacy'),
        ' · ', link('/terms', 'terms'), ' · ', link(CONTACT, 'contact'));
    }
  }
}

// A person's generated mark (R2) as markup for a template, or nothing: `shell.js`
// builds the element, so the drawing is the one both servers' pages use.
function markHtml(mark, size) {
  const drawn = window.memberMark ? window.memberMark(mark, size) : null;
  return drawn ? drawn.outerHTML : '';
}

// Q760: a person's handle as markup, linked to their page (`/players/<handle>`) for a
// member reading the site, so every name on a page is a way to find that person. A
// visitor, the local application, *a former player* (somebody who left has no page) and a
// placeholder such as *somebody* read as plain words. Handles are a narrow set of
// letters, digits, hyphen and underscore, and are encoded anyway.
const FORMER_PLAYER = 'a former player';
function personHtml(handle, mark, size) {
  const name = String(handle || '');
  const drawn = mark ? markHtml(mark, size || 18) : '';
  if (!ACCOUNTS || !VIEWER || !name || name === FORMER_PLAYER || !/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(name)) {
    return drawn + esc(name);
  }
  return `<a href="/players/${encodeURIComponent(name)}">${drawn}${esc(name)}</a>`;
}

// The site's contact (Q720): `web/landing.py`'s `CONTACT`, which `tests/test_landing.py`
// holds equal. A profile page, never an email address.
const CONTACT = 'https://github.com/nabfrew';

// Q731: the header's way to your own page, a button rather than a bare name (a name alone
// read as a label saying who is signed in). The same element as `site/pages.py`'s
// `_member_button`: the mark, the handle, then *your page* and a chevron; on a phone the
// handle goes (`style.py`'s `.memberButton`).
function memberButton(handle, markData) {
  const anchor = link('/profile', '');
  anchor.className = 'memberButton';
  anchor.setAttribute('aria-label', `Your page, signed in as ${handle}`);
  const mark = window.memberMark ? window.memberMark(markData, 20) : null;
  if (mark) anchor.append(mark);
  const name = document.createElement('span');
  name.className = 'memberName';
  name.textContent = handle;
  const go = document.createElement('span');
  go.className = 'memberGo';
  const chevron = document.createElement('span');
  chevron.setAttribute('aria-hidden', 'true');
  chevron.textContent = ' ›';
  go.append('your page', chevron);
  anchor.append(name, go);
  return anchor;
}

function link(where, words) {
  const anchor = document.createElement('a');
  anchor.href = where;
  anchor.textContent = words;
  return anchor;
}

// --------------------------------------------------------------- saying what a word means
//
// This project has its own vocabulary and cannot honestly drop it: an *index* is a
// specific number computed a specific way, and calling it "the score" everywhere would
// make two different things one word. What it can do is never leave a term unexplained
// and never make the explanation the thing you have to read first.
//
// So: the plain sentence is on the screen, and the term is a small (i) beside it. The
// definitions live here because they are about this interface's words. Everything about
// *a measurement* -- what a component is, how it is read -- comes from
// metrics/glossary.py instead, which is the same text the command line prints.
//
// An entry is `[title, text]`, or `[title, text, { more, link }]`: `more` is further
// paragraphs, folded behind *more* in the popover, and `link` is `[href, words]`, the words
// of the text that link there. The site speaks to a visitor in some of these differently;
// `SITE_GLOSS` below holds those, and `glossEntry` is the one reading of a key.
const GLOSS = {
  index: ['The score, out of 1',
    "One number for how interesting a game looks, from measurements taken over a few"
    + " hundred games the computer plays against itself. It is a rough instrument, so"
    + " read it as a band rather than as a place. A game kept here is measured four times"
    + " over four different sets of games and its score is the mean of the four; a game"
    + " somebody has just measured is one run, and every number here says which it is."
    + " The number beside it (+/-) is the sampling error of what is shown. For one run"
    + " that is how much the score moves when that run's own games are resampled, and a"
    + " fresh run usually lands within four of it. For a mean of four it is the error of"
    + " that mean, taken from the four runs' own games. When the library was played again"
    + " from four more seeds, the new mean landed inside it about five times in six, which"
    + " is what an honest +/- should do, because the new mean carries its own error too."],
  // Q647: the chip's (i), followed by the game's own sentence (`trustTag`).
  trust: ['How far to trust it',
    "How much has checked the computer players that measured this score: early estimate"
    + " (only they have), checked (something else agrees with them) or questioned"
    + " (something found more than they see). The score itself does not change."],
  gated: ['Disqualified',
    "Something about this game does not work -- almost every game is drawn, say, or"
    + " thinking harder buys nothing. It is ruled out rather than marked down, because"
    + " those are different verdicts."],
  stale: ['Out of date',
    "It has been measured, but not by the code in this checkout. The engine, the"
    + " opponents or the scoring changed since then. Where the old measurement has a"
    + " number it is still shown, marked provisional, with the scoring rules and the day"
    + " it came from; it is never compared with a current one."],
  // EV2 (`docs/archive/plan-evolving-score.md` §5.1). Not `provisional`, which is a number a
  // designer's browser measured (Q312): this is the library's own last estimate.
  provisionalScore: ['Provisional',
    "The last number this game was given, worked out under an earlier revision of the"
    + " scorer: the opponents, the measurements or the weights have changed since. It"
    + " stays on the card so the library is not blank while the work is owed, and it is"
    + " never current: it has no band, no trust level and no place in a comparison until"
    + " the repair named beside it runs and the estimate is current again."],
  // Drawn only where there is a store, so in a visitor's words on both servers.
  provisional: ['Measured on its designer\'s computer',
    "Your browser can play and score a game itself, so you need not wait for the site."
    + " The site then checks the result: first that every game follows the rules, which"
    + " it scores itself, and later by replaying some of the games to see they come out"
    + " exactly the same. Until then only the person who made the game sees the score;"
    + " once spot-checked, everybody who can see the game does, with that label. It is"
    + " never an official score. Only a score the site measured by playing the games"
    + " itself ranks a game, earns it a badge or puts it forward."],
  unscored: ['Never measured',
    "Nobody has run the tournament for this one. You can start it from the game's own"
    + " page, or run the whole backlog from the panel at the top of the library."],
  // Q953a, `metrics/repeats.py`.
  repeats: ['Different games',
    "A game here is its two players in order and its whole move list. Every player draws"
    + " from its own random stream at every choice, but on some positions no draw changes"
    + " the move, so two seeds can play one game twice. The shares count each different"
    + " game once, over every run, among all the games, the games both of whose players"
    + " search, and the games between the two strongest. Each run is also scored again"
    + " with each game kept once; where that mean is two of the score's sds or more from"
    + " the score, the score rests on repeated games. Repeats lower a score (variety reads"
    + " a replay as a game with a twin), so the number counted once is the higher one. It"
    + " is reported and moves nothing (docs/reference/repeated-games.md)."],
  roster: ['The opponents a score is taken against',
    "Every score comes from the same eight players -- a random mover, a one-move-ahead"
    + " player, and six search budgets -- so that two games' scores mean the same thing."],
  weights: ['Which version of the scoring rules',
    "Which measurements count, and how they are weighed into one number. Versions are"
    + " never edited, only added, so a number always says which rules produced it, and"
    + " its page lists what that version measures. Two versions never share a table."],
  elo: ['How strong, in this game',
    "It uses the chess rating scale, where 100 points is roughly a 64% score and 400"
    + " points about 91%. It is fitted per game, because a search budget is not a"
    + " strength. The same budget is 770 points above random on chess and level with it"
    + " on racing kings."],
  ladderScale: ['The tournament the computer players played here',
    "Every opponent offered on this game played the others, and this is where each one"
    + " finished, on a scale centred on zero: a plus is above the middle of that field and"
    + " a minus below it. It was worked out once and stays where it is, whoever plays"
    + " afterwards. A hundred points is roughly a 64% score and four hundred about 91%."],
  siteScale: ['The scale the people here are rated on',
    "The same strength, written where a person can be put beside it: a new player starts"
    + " at 1500, so an opponent joins at its tournament number plus 1500. This one moves."
    + " Every rated game played here shifts both players, so an opponent people keep"
    + " beating drifts down while its tournament number beside it does not."],
  // The scale people are rated on: the one number a visitor reads for a computer player's
  // strength (`docs/reference/plain-words.md` §3.3; the audit's A5). The site's `siteScale`
  // is this entry.
  ratedHere: ['Rated here',
    "The scale people are rated on. A new player starts at 1500. It moves with every"
    + " rated game, for people and computer players alike."],
  // L2 of `docs/archive/plan-leaderboard.md`: the game page's one table, drawn on the site only.
  leaderboard: ['Leaderboard',
    "This game's one table of ratings: every computer player, and everyone who has"
    + " finished a rated game on it. A rated game moves both players, a person or a"
    + " computer player alike; practice games count for nothing. A new player starts at"
    + " 1500, and a ? beside a number means it is still settling, over about the first ten"
    + " games. Each game has its own table, because being good at one game says nothing"
    + " about another."],
  kind: ['What kind of game it is',
    "Eight things you can read straight off the rules before anybody plays: how big the"
    + " board is, how big the armies are, whether captured pieces come back, how it is"
    + " won, and so on. They exist so a design run can be pointed at a kind of game"
    + " nothing has reached yet."],
  ruleMass: ['Complexity, or how much there is to explain',
    "A count of the moving parts -- piece types, move patterns, special rules -- weighted"
    + " by how much each one adds to what a new player has to be told. Chess is about 52,"
    + " so read it against that, as half of chess or twice it."],
  royal: ['A royal piece',
    "It is the piece the game is about losing, like a king. Whether losing it ends the"
    + " game, and how, is set below under how the game ends, where it is either trapped or"
    + " simply captured. A game with no royal piece is won some other way."],
  rulesetHash: ['The twelve characters that are its name',
    "A game is identified by a hash of its rules -- the board, the pieces, how they move,"
    + " where they start and how it ends -- and by nothing else. Two people who write the"
    + " same game get the same hash, and renaming one changes nothing. Every measurement"
    + " is filed under it, so changing a rule gives you a different game, with no score"
    + " yet."],
  lineage: ['What it came from',
    "Games invented here are revisions of an earlier game, so they form families. This is"
    + " which family, and how many revisions deep."],
  // The words the server sends about two games are `spec/relate.py`'s similarity index.
  // The weighing said here is that module's two constants, `SIMILARITY_RATIO` and
  // `SETTING_WEIGHT`, and `tests/test_web.py` holds the two sentences to them.
  likeness: ['What differs, by kind',
    "The count of rules apart is a count of edits, and every edit counts one: the board,"
    + " the setup, one piece's moves, or one rule of the game. The words marked by kind"
    + " sort what differs into four kinds. Rules are the rules of the game that one of"
    + " the two has and the other does not, such as a hand of captured pieces or a win"
    + " by reaching a square, and a rule both have but set differently counts half."
    + " Mechanisms are ways of moving, such as a slide along a diagonal, that only one"
    + " game's pieces have. Pieces are the kinds of piece only one game has, and squares"
    + " are starting squares held by something different. Each kind weighs eight of the"
    + " kind after it, so eight squares count as one piece, eight pieces as one mechanism"
    + " and eight mechanisms as one rule, and a draw clock or a piece's price counts as"
    + " one square. Games the same number of rules away are listed the most alike first."
    + " A game with nothing within three rules is ordered by a count of fifty-five"
    + " features instead, which picked its relatives better."],
  candidate: ['Not kept yet',
    "It came out of a design run and lives in runs/, which git ignores -- so it lasts"
    + " until the next clean-up unless it is archived or promoted."],
  archived: ['Kept, but not part of the yardstick',
    "It is in the repository under archive/. It is playable, and deliberately not one of"
    + " the games the scoring is calibrated against."],
  committed: ['Part of the yardstick',
    "It is in rulesets/. These are the games the scoring is calibrated against, which is"
    + " why adding one is a decision rather than a save."],
  fitted: ['An opponent that learned what matters here',
    "Instead of assuming material and mobility are what count, this one's priorities were"
    + " fitted from games of this ruleset. It only plays where the fitted version actually"
    + " beat the built-in one, and it says by how much."],
  thumb: ['An opponent with one idea',
    "It is a named plan, such as \'win material\' or \'take the middle\', written as a"
    + " set of priorities. All of them search the same number of positions, so choosing"
    + " between them changes what the opponent wants and nothing else."],
  fairy: ['An engine somebody else wrote',
    "Fairy-Stockfish is an outside chess-variant engine. Where this game's rules have a"
    + " counterpart over there it can be written out as one of its variants and played as"
    + " an opponent, which is the only strength number here that is not this project"
    + " grading itself. The list says whether the rules translate; whether the engine you"
    + " have installed can actually play them is checked on the game's own page, and an"
    + " engine that is not installed is offered nowhere."],
  // Unbeaten, the first game, the highest-rated person, everyone who beat the strongest
  // and a badge are drawn only where there are people, so they speak to a visitor on both
  // servers (A10, A11, A13).
  unbeaten: ['Unbeaten',
    "A game keeps this tag until somebody wins a rated game against its strongest"
    + " computer player. The first person to do it is named on the game for good and the"
    + " rest are counted; a win from either side takes the tag away, a draw does not, and"
    + " practice games count for nothing. Where two computer players are tied as the"
    + " strongest you have to beat each of them. A game whose computer players have never"
    + " been rated has no strongest player, so it carries no tag and is marked unrated."],
  firstPlayed: ['The first game played here',
    "The earliest rated game anybody finished on this game, and whoever was in it. It is"
    + " one per game and it stays put. Practice games count for nothing, and neither do"
    + " games played anywhere but this site, so a game nobody has played here says nothing"
    + " at all."],
  highestPerson: ['The highest-rated person here',
    "Whoever is top of this game's table of people right now, on the same scale the"
    + " computer players are rated on. A rating counts only once it has settled, so"
    + " somebody two games in holds nothing yet, and everybody equal at the top holds it"
    + " together. It changes when a rated game is played here and never because time has"
    + " passed."],
  clears: ['Everyone who has beaten the strongest computer player',
    "Every person who has won a rated game against this game's strongest computer"
    + " player, in the order they did it, with the date and the side they played. A win"
    + " from either side counts and a draw does not; where two computer players are tied"
    + " as the strongest, you have to beat each of them."],
  // A14: *scored*, and *clearly above / clearly below*, on both servers.
  bandClears: ['Clearly above',
    "How many other scored games this one scores above by more than the uncertainty in"
    + " both numbers. Every score is a range rather than a point, so a game is counted"
    + " here only when its whole range sits above the other's. Games scored under"
    + " different scoring rules are never compared."],
  bandClearedBy: ['Clearly below',
    "The same test the other way round: how many scored games score above this one by"
    + " more than the uncertainty in both. A game near the top of the library has few or"
    + " none, and that is the only sense in which anything here is a ranking."],
  bandClose: ['Too close to call',
    "How many scored games this one's range overlaps, so the scores cannot say which of"
    + " the two is better. It is usually the biggest of the three counts, because the"
    + " ranges are wider than most of the gaps between neighbours. That is why no page"
    + " here prints a place like 12th: it would be noise dressed as a ranking."],
  moveRule: ['The rule as written',
    "A piece's moves are read off the game's written rules, and the sentence beside this"
    + " is one of those rules said in words. The rule itself names the steps the piece may"
    + " take, counted sideways and forwards from its own side of the board, and how far"
    + " each one reaches."],
  badge: ['A badge',
    "A dated record of something somebody did here, written once and kept, so it still"
    + " reads true after the numbers behind it have moved. Nothing counts badges and no"
    + " page ranks people by them. Every badge and its rule is listed on the badges page,"
    + " whether or not anybody has earned it. What this one takes is below.",
    { link: ['/badges', 'the badges page'] }],
  designRun: ['A model inventing revisions',
    "It asks a language model for variations on this game, plays a tournament for each"
    + " and keeps what survives. It costs money and takes minutes."],

  // The site's vocabulary (`docs/reference/plain-words.md` §2): a kept word is defined once,
  // here, where a visitor first meets it, and every later word task points its (i) at one
  // of these rather than writing a second definition. Each holds for both servers.
  run: ['A run',
    "One tournament of a few hundred games among the computer players. A library game is"
    + " measured in four runs and its score is the average of the four; a game measured"
    + " once says 1 run."],
  spotChecked: ['Spot-checked',
    "The site replayed some of the games measured on the designer's computer, and they"
    + " came out exactly the same."],
  settling: ['Still settling',
    "A new rating moves a lot with each game, because there is little to go on yet."
    + " Once enough rated games pin it down it is settled, and only a settled rating"
    + " counts in a table of people."],
  // Titled as a kind of game, not as the level word, which only the server spells
  // (`site/stages.py`; `tests/test_site_stages.py`).
  // The library's legend (B2): what the card's first word is, level by level, in
  // `site/stages.py`'s sentences.
  level: ['Where it stands',
    "Every game is at one of five levels. Draft: only people with its link can open it."
    + " On trial: listed for anybody to play and vote on. Shortlisted: members vote on it"
    + " at the next monthly round. Library: scored by the site and listed for everyone."
    + " Baseline: a well-known game every score is measured against, such as chess."],
  baseline: ['A Baseline game',
    "A well-known game every score is measured against, such as chess."],
  practice: ['Practice',
    "Undo is on, and it counts for nothing: not toward anybody's rating, and not toward"
    + " Unbeaten."],
  ratedGame: ['A rated game',
    "A game here that moves both players' ratings, a person's or a computer player's. Only"
    + " a rated game can take away a game's Unbeaten."],
  turns: ['Turns',
    "Each player's move is one turn, so a game of 80 turns is 40 moves each."],
  // R9b (`site/proven.py`): what the game page's *Proven positions* offers.
  provenPositions: ['Proven positions',
    "Positions from games the computer players played here, where the computer searched"
    + " every move and found a forced win for the side to move. Convert one by playing it"
    + " out against hardest: you succeed if the game ends in your win, and a try that does"
    + " not is a practice game you lost, nothing more. Where only one move wins, you can"
    + " also find that move; a move the search could not settle is said to be unsettled,"
    + " never wrong."],
  strongest: ['The strongest computer player',
    "The computer player rated highest on this game. Beating it in a rated game is what"
    + " takes away the game's Unbeaten."],
  nominate: ['Nominate',
    "Put it forward for this month's members' vote."],
  community: ['A community',
    "Games more like each other than like the rest."],
  // A family's name on the score panel opens this, with the family's own question from
  // `metrics/glossary.py` as the (i)'s extra (G27), so a tap reads what a hover does.
  family: ['What this part of the score asks',
    "The score multiplies these together, so a game near zero on one of them scores low"
    + " however well it does on the rest. This one asks:"],
  playability: ['The playability check',
    "Three quick tests a draft passes before the site keeps it: its rules are a complete"
    + " game, a game of it plays and comes to an end, and a computer player that thinks ahead can"
    + " win it."],
  // Q838c (`docs/archive/plan-portal-look.md` §5): the legend's (i) for a portal. The popover draws
  // four ends (`figure`, `viz/wrapped.js`'s `portalKeyFigure`, as the board draws them), each
  // with its sentence; the text here is the same reading in words, for a hover and a reader
  // who cannot see the drawing.
  portalKey: ['How to read a portal',
    "What a portal stands on is who may move it: a white token White, a black token Black, a"
    + " half-and-half token either side, and one sunk into the board nobody. Its arms are whose"
    + " pieces may go in: white arms White's, black arms Black's, one of each both sides'. One"
    + " with no arms lets nobody in: it is only a way out. The same colour and number are one"
    + " group, and the beak on an end's rim points where you come out: a piece goes in at one"
    + " end and comes out one step beyond the end its beak points at, in the direction it was"
    + " going. Several beaks may point at one way out. Dashed arms let only some pieces in.",
    { figure: () => portalKeyFigure(typeof SESSION !== 'undefined' && SESSION && SESSION.ruleset
      ? SESSION.ruleset.sides : null) }],
  // Q843d, revised by Q867c (`docs/archive/plan-portals-2.md` §5.3R): the legend's (i) for a fold
  // portal, drawn as the board draws it (`viz/wrapped.js`'s `foldKeyFigure`): a pool with a
  // piece in it, a pair whose beaks and map arrows say where a piece comes out, then its bands
  // and its collar. The text is the same reading in words.
  foldKey: ['How to read a fold portal',
    "A fold is a pool of light lying in the board. A piece can stand in it, and is taken there"
    + " as anywhere. A piece that ends its move on an empty fold may stay there, or come out on"
    + " the end the fold's beak points at, taking a piece there as any move does; a move that"
    + " passes over a fold is not changed. Its bands are whose pieces may go through: white White's,"
    + " black Black's, one of each both sides', plain nobody's. Its collar is who may move"
    + " it, even from under a piece: none nobody, white White, black Black, half and half either"
    + " side. The same colour and number are one group, and the portal map draws where each"
    + " end leads as arrows.",
    { figure: () => {
      const rules = typeof SESSION !== 'undefined' && SESSION && SESSION.ruleset ? SESSION.ruleset : null;
      // The game's first piece that is no portal end, as its first side draws it.
      const ends = new Set(((rules && rules.portals && rules.portals.ends) || []).map(end => end.type));
      const icon = rules ? ((rules.icons || []).find((pair, type) => !ends.has(type) && pair && pair[0])
        || [])[0] : null;
      return foldKeyFigure(rules ? rules.sides : null, icon);
    } }],
};

// What the site says instead, where the local application's sentence is the maintainer's
// and stays (`docs/archive/plan-understanding.md` §3.3, rows A1-A9, A12). Same keys, same shape;
// `glossEntry` reads this first where `ACCOUNTS` says this is the site.
const SITE_GLOSS = {
  // A1: §2.1's three paragraphs, the first shown and two behind *more*.
  index: ['The score, out of 1',
    "It says how interesting a game looks when the computer plays it."
    + " Computer players of several strengths play the game against each other hundreds of"
    + " times. A game scores well when the stronger player usually wins (so skill pays),"
    + " the first player has no big edge, games don't stall or end in a few moves, there"
    + " are real choices on most turns, and no two games go the same way. Six qualities"
    + " are scored out of 1 and multiplied, so a game that fails at one of them can't make"
    + " up for it elsewhere.",
    { more: [
      "Every score says what it rests on. A library game is measured four times, on four"
      + " different sets of games, and its score is the average: 0.810 · 4 runs. A"
      + " game measured once says 1 run. The ± is how far the number could move from"
      + " chance alone. Two games whose ranges overlap are too close to call, which is why"
      + " the site never prints a place like 12th. When the computer players or the scoring"
      + " change, an old score is not shown as current: it reads out of date until it is"
      + " measured again.",
      "It is the best estimate available today, not a verdict. The computer players are"
      + " strong on games like chess and weaker on games won by a long plan, where they can"
      + " miss what a person sees. The score is measured again when stronger computer"
      + " players arrive, and people playing here is how the site finds out where they fall"
      + " short: if you beat the strongest computer player easily, that is news about the"
      + " score too.",
    ] }],
  repeats: ['Different games',
    "The computer players make some of their choices at random, but in some games those"
    + " choices change nothing and they play the same game again, move for move. This"
    + " counts how many of a game's games were different. When the score,"
    + " worked out again with each game counted once, comes out more than twice its ±"
    + " away, the score rests on repeated games: it says more about the computer players'"
    + " habits than about the game. Repeats lower a score, so the figure counted once is"
    + " the higher one."],
  gated: ['Ruled out',
    "It failed a basic test of a working game, such as games that never finish, or a"
    + " player who thinks being no better than one who moves at random. It is ruled out"
    + " rather than marked down, because a game that does not work is not a weak one."],
  // A2.
  stale: ['Out of date',
    "Since it was measured, the computer players or the scoring have changed. Its last"
    + " score is shown marked provisional where it has one, and it is not compared with"
    + " other games' scores. It will be measured again."],
  // EV2: the library's last estimate while it is measured again. A score is the best
  // estimate the site has today, and it moves as the computer players and the scoring
  // improve (`docs/archive/plan-evolving-score.md` §0).
  provisionalScore: ['Provisional',
    "This is the game's last score, worked out before the latest change to how games are"
    + " scored. It is the best estimate there is until the game is measured again, which"
    + " happens on its own. Until then it is not compared with other games, so it says"
    + " nothing about which of two games is clearly better, and how far to trust it is"
    + " not shown."],
  // A3.
  roster: ['The computer players',
    "Every score comes from the same computer players, from one that moves at random to"
    + " ones that look at many positions before each move, so two games' scores mean the"
    + " same thing."],
  weights: ['Which version of the scoring rules',
    "The scoring rules say what is measured and how much each part counts. They have"
    + " numbered versions, and a change is a new version rather than an edit, so every"
    + " score says which version produced it. Scores under different versions are never"
    + " compared."],
  // A4.
  elo: ['How strong, on this game',
    "A rating is worked out separately for each game: the same computer player is strong"
    + " at chess and no better than random at racing kings. 100 points is about a 64%"
    + " score."],
  // A6.
  kind: ['What kind of game it is',
    "Eight things you can read straight off the rules before anybody plays: how big the"
    + " board is, how big the armies are, whether captured pieces come back, how it is"
    + " won, and so on. They let you find games of one kind, and show which kinds nobody"
    + " has made yet."],
  // A8.
  likeness: ['What differs, by kind',
    "Rules count most, then ways of moving, then kinds of piece, then starting squares."
    + " The fewer the differences, the more alike."],
  // A9.
  fairy: ['An engine written by other people',
    "Fairy-Stockfish is a chess-variant engine written by other people. Where it can play"
    + " a game, you can play against it on that game's page."],
  // A12: the site draws no rule beside it, so it does not describe one.
  moveRule: ['The rule as written',
    "A piece's moves are read off the game's written rules, and the sentence beside this"
    + " is one of them said in words."],
};
// A5: one scale for a visitor, `siteScale` being `ratedHere`. The tournament's own column
// and its (i), `ladderScale`, are the local application's alone since W3 (§2.3, G14).
SITE_GLOSS.siteScale = GLOSS.ratedHere;

// The one reading of an (i)'s key: the site's words where this is the site and it has
// some, the shared entry otherwise.
function glossEntry(key) {
  return (ACCOUNTS && SITE_GLOSS[key]) || GLOSS[key] || null;
}

// The (i) itself. Deliberately tiny and skippable -- the sentence beside it is the
// answer, and this is for the reader who wants the rest. `title` carries the same text,
// so a hover works before anything is clicked and a browser with no JavaScript still
// says it.
//
// `extra` is what this one (i) knows that the glossary does not, such as why a
// particular score is out of date. It travels on the button as well as in the title,
// because a title is a hover and a phone has no hover, so the reason was on the site
// and unreadable there.
function info(key, extra) {
  const bits = infoBits(key, extra);
  if (!bits) return '';
  return `<button class="info" data-gloss="${esc(key)}"${
    bits.more ? ` data-more="${esc(bits.more)}"` : ''} title="${esc(bits.title)}"
    aria-label="what ${esc(bits.label)} means">i</button>`;
}

// What the (i) is, before it is either markup or a node. One reading, so the two
// spellings of the same button cannot come to carry different text.
function infoBits(key, extra) {
  const entry = glossEntry(key);
  if (!entry) return null;
  const more = extra ? String(extra).replace(/\s+/g, ' ') : '';
  const body = (entry[1] + (more ? ' ' + more : '')).replace(/\s+/g, ' ');
  return { label: entry[0], more, title: entry[0] + ': ' + body };
}

// The same (i) as an element, for a view that builds its lines out of text nodes rather
// than out of markup. The click handler is delegated on the document and reads the two
// data attributes, so a button built here behaves exactly as one written as markup.
function infoNode(key, extra) {
  const bits = infoBits(key, extra);
  if (!bits) return null;
  const button = document.createElement('button');
  button.className = 'info';
  button.dataset.gloss = key;
  if (bits.more) button.dataset.more = bits.more;
  button.title = bits.title;
  button.setAttribute('aria-label', 'what ' + bits.label + ' means');
  button.textContent = 'i';
  return button;
}

// ------------------------------------------------- where a game's score sits in the library
//
// Three counts, computed on the server over every measured game and handed here whole.
// This draws them and works nothing out: comparing two scores is the server's act, and a
// page that did its own arithmetic over two intervals would be a second opinion about
// what the measurement says.
//
// One function, drawn on the card and on the game's page, so the two cannot disagree.
// `band` is absent whenever the measurement is stale or missing, and then nothing is
// drawn at all -- there is no *unplaced* to show, because an unreadable score is not a
// low one.
function bandHtml(band) {
  if (!band) return '';
  return `<div class="muted small" style="margin-top:7px">Against the other ${
    band.of} scored game${
    band.of === 1 ? '' : 's'}: clearly above ${band.clears}${info('bandClears')}, clearly
    below ${band.clearedBy}${info('bandClearedBy')}, too close to call with ${band.close}${
    info('bandClose')}.</div>`;
}

// ------------------------------------------------------------- where a game stands
//
// Q503b. A game's level is one word, decided by the server in one place
// (`site/stages.py`: `stage` from the row on the site, `local` from where the file is on
// a checkout) and sent as `level`, with `track`, the designer's four steps with the
// current one lit. The page draws what it is sent and holds no level word of its own, so
// the card, the game page and the designer cannot name one game two ways.
function trackHtml(track) {
  if (!track || !track.length) return '';
  return `<ol class="track">${track.map(step => `<li class="${step.lit ? 'lit' : ''}"
    >${esc(step.word)}</li>`).join('')}</ol>`;
}

// The card's first pill: the level word, and nothing where the listing sent none.
// Its (i) where the server names one for the level (`stages.GLOSSES`: Baseline, B23).
function levelTag(game) {
  // A preview's level (Q931) is drawn as its badge, louder than the other level words:
  // it is the one that says how to read everything else on the card.
  return game.level ? `<span class="tag level${game.preview ? ' preview' : ''}">${
    esc(game.level)}</span>${game.levelGloss ? info(game.levelGloss) : ''}` : '';
}

// A preview (Q931): a new kind of game, still being tested, listed apart from the library
// and in no scored surface. The sentence is the spec's own `provenance.preview`, served as
// `preview` on its card and its page by `library.preview`; the page writes none of it.
// Where a game is a preview its card shows this in place of the score, and its page shows
// it under the name and above the score.
function previewHtml(said) {
  return said ? `<div class="previewNote">${esc(said)}</div>` : '';
}

// A game kept on purpose for what it shows says so beside its number, wherever the number
// is, so nobody reads the number as a score -- the collection keeps one game as an example
// of a game that cannot really be played (the user's decision of 2026-09-09, Q141). The
// sentence is the spec's own `provenance.exhibit`, served as `exhibit` on a card, a game,
// a neighbour and a map node by `library.exhibit`; the page writes none of it and names
// no game. On the face rather than in a hover, because a phone has no hover.
function exhibitHtml(said) {
  return said ? `<div class="exhibit">${esc(said)}</div>` : '';
}

// How far to trust a score (Q647, `docs/archive/plan-confidence.md` §2.1). The server reads the
// evidence and words it (`trust.py`: `card`, `shown`); the page draws what it is sent and
// decides nothing. A card carries the level word as a chip beside the score, its sentence
// in the (i); a game page carries the line under the score: *How far to trust it*, the
// word, one sentence, and behind *more* each piece of evidence with at most one plain
// count and a date, and each earlier score. The level is a word only: no count is on the
// line. `numbers` is the maintainer's (the local application, and an admin on the site;
// a visitor is sent none). No level, no line: a score out of date says so itself.
//
// EV2: a provisional estimate has no trust level, since trust reads evidence against the
// current basis and a provisional estimate has none (§5.1). The server already sends none;
// `status` withholds it here too, so a stale payload can never draw one.
function trustTag(trust, status) {
  if (status === 'provisional') return '';
  return trust && trust.level
    ? `<span class="tag trustLevel">${esc(trust.level)}</span>${info('trust', trust.said)}` : '';
}

function trustHtml(trust, status) {
  if (status === 'provisional' || !trust || !trust.level) return '';
  const more = trust.more || [];
  const numbers = trust.numbers || [];
  return `<div class="trust"><strong>How far to trust it: ${esc(trust.level)}.</strong> ${
    esc(trust.said)}${more.length ? `
    <details class="trustMore"><summary>more</summary>
      <ul class="rules small">${more.map(line => `<li>${esc(line)}</li>`).join('')}</ul>
    </details>` : ''}${numbers.length ? `
    <details class="trustMore"><summary class="muted">the numbers</summary>
      <ul class="rules muted small">${numbers.map(line => `<li>${esc(line)}</li>`).join('')}</ul>
    </details>` : ''}</div>`;
}

// ------------------------------------------------------ how many runs a number rests on
//
// A run is one tournament from one seed. A game the library keeps is measured from four
// seeds and its score is the mean of the four; a game somebody has just measured is one
// run. The two are the same shape and not the same claim -- a one-run interval is about
// twice as wide -- so the count goes beside every number rather than being left for a
// reader to infer from a seed list.
//
// One place, because a card saying "4 runs" next to a page saying "one run" about the
// same measurement is the failure this whole set of fields exists to stop.

// To twenty, because the score panel counts a version's components with it too, and
// versions have measured sixteen.
const RUN_WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                   'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
                   'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'];

function runCount(seeds) {
  return Array.isArray(seeds) ? seeds.length : 0;
}

// A small count as a sentence spells it. Prose that says "the 4 runs" beside "four runs"
// reads as two different facts about the same measurement.
function countWord(n) {
  return RUN_WORDS[n] || String(n);
}

// `one run` or `mean of four runs`, for a sentence. Empty when nothing says how many,
// which is what an unmeasured game and an older file both look like.
function runsPhrase(seeds) {
  const n = runCount(seeds);
  if (!n) return '';
  // *average* on the site (B10, §3.2's *run*); the local application keeps *mean*.
  return n === 1 ? 'one run' : (ACCOUNTS ? 'average of ' : 'mean of ') + countWord(n) + ' runs';
}

// Which scoring rules a number was taken under, as a sentence says it: *scoring rules
// version 11* for the stored `v11` (G22), the badges' own wording. A version spelled any
// other way is printed as it came.
// ---------------------------------------------------------- a provisional estimate
//
// EV2 (`docs/archive/plan-evolving-score.md` §5.1). A score whose basis no longer matches the
// current scorer is drawn with its number, in the same place, with this mark and one
// sentence naming the revision it came from and its day -- never bare, never as current.
// `estimate` is a card (`status`, `revision`, `measuredAt`) or a score (`status`,
// `revision`, `measured_at`); both spellings are read so one sentence serves both.
const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
  'August', 'September', 'October', 'November', 'December'];

function dayWords(iso) {
  const found = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso || ''));
  return found ? `${Number(found[3])} ${MONTH_NAMES[Number(found[2]) - 1]} ${found[1]}` : '';
}

function isProvisional(estimate) {
  return !!estimate && estimate.status === 'provisional';
}

function provisionalSentence(estimate) {
  if (!isProvisional(estimate)) return '';
  const revision = estimate.revision || estimate.weights || estimate.weights_version;
  const day = dayWords(estimate.measuredAt || estimate.measured_at);
  const basis = [revision ? rulesVersion(revision) : '', day].filter(Boolean).join(', ');
  return `Provisional: worked out before the latest change to how games are scored${
    basis ? ` (${basis})` : ''}. It will be measured again.`;
}

// The mark beside a provisional number: the word and its (i), then the sentence.
function provisionalMarkHtml(estimate) {
  if (!isProvisional(estimate)) return '';
  return `<div class="small provisionalMark"><span class="tag stale">provisional</span>${
    info('provisionalScore')} ${esc(provisionalSentence(estimate))}</div>`;
}

function rulesVersion(version) {
  const found = /^v(\d+)$/.exec(String(version || ''));
  return 'scoring rules ' + (found ? 'version ' + found[1] : String(version || ''));
}

// The same fact in a pill, where digits scan and words do not.
function runsShort(seeds) {
  const n = runCount(seeds);
  return n ? n + ' run' + (n === 1 ? '' : 's') : '';
}

// How the runs split, when they did not agree: the split itself and then each refusing
// run's own words, named by seed. Both sentences are the server's (`metrics/seeds.py`)
// and neither is assembled here from the two counts. Empty when the runs agreed or there
// was only one, which is the case a reader does not need told.
//
// A game the vote ruled out carries the same sentences in `violations` already, so a
// caller that draws those draws this only where that list is empty.
//
// A run is named by its seed on the local application, where the seed is how the maintainer
// finds its directory, and by its place among the runs on the site, where a seed means
// nothing (A15). The site reads each sentence in a visitor's words where the server sent
// them (`web/sitescore.py`, G30): `site` beside the measurement's own.
function splitSentences(verdict, runs) {
  if (!verdict || !verdict.split) return [];
  const said = [siteSaid(verdict, 'split') || verdict.split];
  const all = runs || [];
  all.forEach((run, at) => {
    if (run.viable === false) {
      const named = ACCOUNTS ? `run ${at + 1} of ${all.length}` : 'run ' + run.seed;
      for (const why of siteSaid(run, 'violations') || run.violations || []) {
        said.push(named + ': ' + why);
      }
    }
  });
  return said;
}

// What the site says for one of a measurement's sentences: its `site` words where this is
// the site and the server sent some, else nothing, and the caller draws the measurement's.
function siteSaid(holder, key) {
  return (ACCOUNTS && holder && holder.site && holder.site[key]) || null;
}

// How far a ranked game leans toward one seat, in a sentence, or nothing.
//
// The seat check refuses a game one seat wins nearly every time and tolerates a share of
// two in three below that. This is the gap between the two: a game whose strongest
// players favour one seat past what the check tolerates, without the games to put it
// beyond doubt, is ranked with this printed beside its number. It refuses nothing, and a
// game the check *did* refuse says so in its violations instead.
function seatLeanText(score) {
  const seat = score && score.seat;
  if (!seat || seat.refuses || !seat.lean) return '';
  return siteSaid(seat, 'lean') || seat.lean;
}

// What the popover says, which is the glossary's sentence, then whatever this one (i)
// knows on top of it, then the entry's further paragraphs folded behind *more*. Its own
// function so the handler below is about positioning.
function glossBubble(key, more) {
  const entry = glossEntry(key);
  if (!entry) return '';
  const { more: further = [], link, figure } = entry[2] || {};
  const said = text => {
    const words = esc(String(text).replace(/\s+/g, ' '));
    return link ? words.replace(esc(link[1]),
      `<a href="${esc(link[0])}">${esc(link[1])}</a>`) : words;
  };
  // A `figure` is drawn markup (the portal's four ends) shown in place of the text, which
  // stays the button's `title`; it is the page's own drawing, never anything a person wrote.
  if (figure) return `<b>${esc(entry[0])}</b>${figure()}`;
  return `<b>${esc(entry[0])}</b><p>${said(entry[1])}</p>${
    more ? `<p>${esc(more)}</p>` : ''}${further.length ? `<details><summary>more</summary>${
    further.map(text => `<p>${said(text)}</p>`).join('')}</details>` : ''}`;
}

// One popover for the whole page, moved to whatever was clicked. A panel per term would
// be a hundred hidden divs; a `title` alone waits a second and cannot be read on a touch
// screen.
function wireInfo() {
  if (wireInfo.done) return;
  wireInfo.done = true;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.hidden = true;
  document.body.appendChild(bubble);
  // It is positioned in page coordinates, so a scroll would carry it away from the (i)
  // it belongs to. Escape closes it because it is, briefly, the thing in front.
  window.addEventListener('scroll', () => { bubble.hidden = true; }, { passive: true });
  window.addEventListener('keydown', event => {
    if (event.key === 'Escape') bubble.hidden = true;
  });
  document.addEventListener('click', event => {
    // Inside the popover is reading it: opening *more*, or following its link.
    if (bubble.contains(event.target)) return;
    const button = event.target.closest('[data-gloss]');
    if (!button) { bubble.hidden = true; return; }
    event.preventDefault();
    event.stopPropagation();
    const entry = glossEntry(button.dataset.gloss);
    if (!entry) return;
    // Two (i)s can share a term and carry different `extra`, so what the bubble is
    // currently showing is the pair and not the term alone.
    const more = button.dataset.more || '';
    show(button, button.dataset.gloss + '\n' + more, () => glossBubble(button.dataset.gloss, more));
  });
  // The first-visit study's F3: a card's tags say what they mean in a `title`, and the line
  // above the cards says *tap or point at one*. A phone has no pointer to rest, and a tap
  // on a tag was a tap on its card, which opened the game. So where nothing can hover, a
  // tap on a tag opens its sentence here instead, ahead of the card's own click (capture).
  document.addEventListener('click', event => {
    if (!cannotHover() || bubble.contains(event.target)) return;
    const tag = event.target.closest('.tag[title]');
    if (!tag || !tag.title) return;
    event.preventDefault();
    event.stopPropagation();
    show(tag, 'tag\n' + tag.title, () => `<p>${esc(tag.title)}</p>`);
  }, true);

  function show(anchor, showing, html) {
    if (!bubble.hidden && bubble.dataset.for === showing) { bubble.hidden = true; return; }
    bubble.dataset.for = showing;
    bubble.innerHTML = html();
    bubble.hidden = false;
    const box = anchor.getBoundingClientRect();
    const width = Math.min(320, window.innerWidth - 24);
    bubble.style.width = width + 'px';
    bubble.style.left = Math.max(12, Math.min(box.left, window.innerWidth - width - 12)) + 'px';
    bubble.style.top = (box.bottom + window.scrollY + 6) + 'px';
    // Above the (i) where it would run off the foot of the screen and fits above (Q838c: the
    // portal's drawn key, under a legend near the foot of the page, was cut off, and
    // scrolling to it closes it).
    const tall = bubble.offsetHeight || 0;
    if (box.bottom + 6 + tall > window.innerHeight && box.top - 6 - tall >= 0) {
      bubble.style.top = (box.top + window.scrollY - 6 - tall) + 'px';
    }
  }
}

// A screen nothing can hover over: a `title` there is never read.
function cannotHover() {
  return !!(window.matchMedia && window.matchMedia('(hover: none)').matches);
}

// A game's address, to send to somebody (the first-visit study's B3). A saved draft is
// listed nowhere, so its link is the one way anybody else reaches it, and two designers
// found no way to share one: the page had no control for it. The button carries the
// hash and nothing else; the address is this page's own origin and the router's
// `#/game/` route, the one every link to a game uses.
function gameLink(hash) {
  return location.origin + '/#/game/' + hash;
}

function copyLinkHtml(hash) {
  return `<button data-copy-link="${esc(hash)}"
    title="copy this game's address, to send to somebody">copy link</button>`;
}

// One handler for every copy button the page draws, wherever it is drawn and however
// often its panel is redrawn. Where the browser will not write to the clipboard -- an
// address that is not https, or a refusal -- the link is put beside the button,
// selected, to copy by hand: a button that silently does nothing is worse than none.
function wireCopyLinks() {
  if (wireCopyLinks.done) return;
  wireCopyLinks.done = true;
  document.addEventListener('click', event => {
    const button = event.target.closest && event.target.closest('[data-copy-link]');
    if (!button) return;
    const link = gameLink(button.dataset.copyLink);
    const byHand = () => {
      const shown = button.nextElementSibling;
      const box = shown && shown.dataset && shown.dataset.copied === 'yes'
        ? shown : document.createElement('input');
      box.dataset.copied = 'yes';
      box.readOnly = true;
      box.value = link;
      box.style.maxWidth = '100%';
      if (box !== shown) button.insertAdjacentElement('afterend', box);
      box.select();
    };
    const writing = navigator.clipboard && navigator.clipboard.writeText
      ? navigator.clipboard.writeText(link) : Promise.reject(new Error('no clipboard'));
    writing.then(() => {
      button.textContent = 'link copied';
      setTimeout(() => { button.textContent = 'copy link'; }, 2500);
    }, byHand);
  });
}

function stopPolling() {
  if (TIMER) { clearInterval(TIMER); TIMER = null; }
  if (RATING_TIMER) { clearInterval(RATING_TIMER); RATING_TIMER = null; }
  if (TEST_TIMER) { clearInterval(TEST_TIMER); TEST_TIMER = null; }
  if (SWEEP_TIMER) { clearInterval(SWEEP_TIMER); SWEEP_TIMER = null; }
}

// ------------------------------------------------------------------- the route
//
// The hash is the single source of truth about which view is showing, so a view never
// calls another one -- it navigates, and the router draws. That is what makes Back work
// and what makes a game linkable, neither of which it was: every screen was one URL and
// the browser's own buttons did nothing.
//
// A game in progress is in this tab's memory while you are playing it, and in the store
// for as long as the row runs. So `#/play/<id>` is a real address: the tab's own session
// answers it without a request, and every other way in -- a reload, a bookmark, a *carry
// on* link -- fetches the position and draws it.
//
// The local application keeps its sessions in memory alone, so a reload there finds
// nothing and the route answers 404. That and a game the site will not open both land in
// the library with the server's own sentence on screen.
function go(hash) {
  if (location.hash === hash) render();
  else location.hash = hash;
}

// Q723: the page answers two addresses. `/` is the landing page and `/game` the games, for
// everybody, signed in or not; every hash route works on either, so a link made before
// `/game` existed (`/#/game/<hash>`, `/#/play/<id>`, a bookmark) opens what it named.
// `#/` is the games wherever it is read, and is written down as `/game`, which is how a
// way back to the library (*back*, a deleted game, a saved `/#/`) arrives there.
const GAMES_PATH = '/game';

function onGamesPath() {
  return location.pathname.replace(/\/+$/, '') === GAMES_PATH;
}

// To an address of the page's own (`/` or `/game`) without loading the page again.
function goPath(path) {
  if (location.pathname + location.hash !== path) history.pushState(null, '', path);
  render();
}

// Back and Forward between `/` and `/game` change no hash, so no `hashchange` comes and the
// `popstate` that does draws. A step that changes the hash is the `hashchange`'s to draw.
let RENDERED = '', RENDERED_HASH = '';
function onHistory() {
  if (location.hash === RENDERED_HASH && location.href !== RENDERED) render();
}

function render() {
  wireInfo();
  wireCopyLinks();
  // An error belongs to the screen that raised it, and this is a different screen. The
  // boxes shown beside a control go with the view that held them; the fixed one is
  // outside `view` and would otherwise outlive its page.
  clearToast();
  // The board alone (Q638, `board.js`) is the board's view; any page drawn leaves it. A page
  // that loads this file without the board's (the landing page's tests) has no view to leave.
  if (typeof setBoardFull === 'function') setBoardFull(false, 'route');
  document.getElementById('tagline').textContent = TAGLINE_SERVED;
  // A new view starts at its own top. The page it replaces kept the window's scroll, so
  // a game opened from the bottom of the library arrived scrolled to its middle -- past
  // its name, past the play panel, at whatever paragraph the old page had reached. Only
  // the window scrolls; the panels that scroll inside themselves, such as the board's
  // move list, are untouched.
  window.scrollTo(0, 0);
  const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  // Q723: `#/` is the games, said as `/game` (a saved `/#/` is replaced, not followed).
  if (!parts.length && location.hash.replace(/^#/, '') === '/') {
    try { history.replaceState(null, '', GAMES_PATH); } catch (error) { /* stays */ }
  }
  RENDERED = location.href;
  RENDERED_HASH = location.hash;
  // Q718, Q723: the landing page, for everybody at the bare address and at `#/about`
  // anywhere. The games are `/game`, and `#/` wherever it is read.
  if (parts[0] === 'about' || (!parts.length && !onGamesPath()
      && location.hash.replace(/^#/, '') !== '/')) {
    return showLanding().catch(fail);
  }
  if (parts[0] === 'game' && parts[1]) return showGame(parts[1]).catch(fail);
  // The library's second view: the same games as a map, drawn by the graph library and
  // steered. A place rather than a way of looking, unlike the filters, because it is a
  // page somebody links to. `#/map/explore` is where it was built and still leads here.
  if (parts[0] === 'map') return showExplorer().catch(fail);
  // The changelog, which both servers hold and neither needs an account for.
  if (parts[0] === 'whats-new') return showWhatsNew().catch(fail);
  // The monthly round (Q503h): the latest, or one by the 1st of its month. The site's.
  if (parts[0] === 'round') return showRound(parts[1] || '').catch(fail);
  if (parts[0] === 'play' && parts[1]) {
    if (SESSION && SESSION.id === parts[1]) return showBoard(SESSION);
    pending('Loading the game…');
    return api('/api/session/' + parts[1]).then(showBoard)
      .catch(error => showLibrary().then(() => fail(error)).catch(fail));
  }
  // The designer takes an optional ruleset to start from, which is what "make one like
  // this" on a game's page links to. With none it starts from a template.
  if (parts[0] === 'design') return showDesigner(parts[1] || '').catch(fail);
  if (parts[0] === 'opponent' && parts[1]) return showBotBuilder(parts[1]).catch(fail);
  if (parts[0] === 'run' && parts[1] && RUN && RUN.id === parts[1]) return showRun(RUN);
  return showLibrary().catch(fail);
}
