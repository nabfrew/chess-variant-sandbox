// The landing page (Q718): what everybody meets at `/`, signed in or not (Q723); the games
// are at `/game`.
//
// The words are the server's: `web/landing.py` draws the page (LP4, direction B of
// `docs/plan-landing.md`: the question and its turning endings, then one story start to
// finish, then the wall of words) from the stories `web/stories.py` rendered, and the single
// page carries it as `<template id="landingPage">`. This file clones it into the view, hides
// what this server has not got (`data-needs`), draws the hero's games from the listing, and
// plays the story's two games from `/api/stories` with the board motion every page shares
// (`motion.js`). It draws no sentence of its own beyond a move's words and a hand's count,
// and works out nothing about any game (rule 3).

async function showLanding() {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  const template = document.getElementById('landingPage');
  view.textContent = '';
  view.append(chooseStory(template.content.cloneNode(true)));
  // The landing page is in none of the header's sections, at `/` or at `#/about`.
  document.querySelectorAll('#sections a').forEach(link => link.removeAttribute('aria-current'));
  dropUnoffered(view);
  wireWallDoors();
  wireWallRails();
  // The hero's boards and today's game wait for the listing; the words and the doors do not.
  const listing = api('/api/library').catch(() => null);
  wirePlayDoors(view, listing);
  startJourney();
  // LP6: a reload or a step back to the same story comes back where the reader was in it
  // (kept by the head's `story.js`; the router has just put the view at its top).
  const back = typeof window.landingScroll === 'function' ? window.landingScroll() : null;
  // LP11: the play screen's way back to the story lands on its step.
  const beat = LANDING_BEAT && view.querySelector(`#beats [data-beat="${LANDING_BEAT}"]`);
  LANDING_BEAT = null;
  if (beat) beat.scrollIntoView({ block: 'start' });
  else if (back) window.scrollTo(0, back);
  const data = await listing;
  if (!data) return;
  takePieceArt(data);
  const today = view.querySelector('[data-today]');
  if (today && data.firstStep && data.firstStep.hash) today.href = gamePath(data.firstStep.hash);
  // A server with no day's game (the local application) has only the list behind *Play today's
  // game*; beside a story's own door, which starts a board, it goes.
  else if (today && view.querySelector('.landingWays a[data-play]')) today.hidden = true;
  startHero(data);
}

// LP9 (`docs/reference/landing-cold-read.md` §3, R6): every door to play on the landing lands
// on a started board. A story's door (`data-play`, `landing.play_door`) starts a practice game
// of its fixed game against the gentlest computer player the game's own page offers, the
// visitor in the story's seat (`data-seat`); *Play today's game* starts the day's game against
// the gentlest the server named for it (`firstStep`, as the games page's own button does). A
// press asking for a new tab, and a game with no opponent to seat, follow the link instead: the
// game's page, with the seat chosen (`?seat=`, `detail.js`). The page picks no move and no
// game; the opponents are the server's and the session is the core's (rule 3).
function wirePlayDoors(root, listing) {
  const plain = event => !(event.button || event.metaKey || event.ctrlKey || event.shiftKey
    || event.altKey);
  root.querySelectorAll('a[data-play]').forEach(door => {
    door.addEventListener('click', event => {
      if (!plain(event)) return;
      event.preventDefault();
      // LP11: the game remembers the story that started it, for the play screen's way back.
      const part = door.closest('.storyPart');
      if (part) keepStoryGame(part.dataset.story, door.dataset.play);
      playNow(door, door.dataset.play, Number(door.dataset.seat) || 0, door.dataset.agent);
    });
  });
  wireHeroBoard(root);
  const today = root.querySelector('[data-today]');
  if (!today) return;
  today.addEventListener('click', async event => {
    if (!plain(event)) return;
    event.preventDefault();
    pending('Starting the game…');
    const data = await listing;
    const step = data && data.firstStep;
    const gentle = step && step.opponents && step.opponents.gentle;
    if (step && step.hash && gentle) return playNow(today, step.hash, 0, gentle.agent);
    clearPending();
    goPath(step && step.hash ? gamePath(step.hash) : GAMES_PATH);
  });
}

// LP11 (`landing-cold-read.md` §6.2 item 2): the hero's board looked playable at 375 and was
// a replay, so a tap on the story's own board takes the story's door beside it: the same game,
// started. The other endings' boards are other games, and stay pictures.
function wireHeroBoard(root) {
  const part = root.querySelector('.storyPart');
  const door = part && part.querySelector('.landingWays a[data-play]');
  const state = door && part.querySelector(`.heroState[data-story="${part.dataset.story}"]`);
  const board = state && state.querySelector('[data-board]');
  if (!board) return;
  board.classList.add('heroPlays');
  board.setAttribute('role', 'button');
  board.setAttribute('tabindex', '0');
  board.setAttribute('aria-label', door.textContent);
  board.title = door.textContent;
  board.addEventListener('click', () => door.click());
  board.addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    door.click();
  });
}

// LP11 (§6.2 item 1): a game a story's door started links back to the story (the play screen's
// *← the story* and its ending's *how this game was made*, `showBoard` and `drawEnded`). What
// the tab keeps is the story and its fixed game, so a new game of the same game, and a reload,
// still find the way back; storage refused keeps it for this page only.
const STORY_GAME_KEY = 'storyGame';
let STORY_GAME = null;

function keepStoryGame(key, hash) {
  if (!key || !hash) return;
  STORY_GAME = { key: key, hash: hash };
  try { window.sessionStorage.setItem(STORY_GAME_KEY, key + ' ' + hash); } catch (refused) { /* this page */ }
}

// The story a game of `hash` was started from in this tab, or null.
function storyGameFor(hash) {
  let kept = STORY_GAME;
  if (!kept) {
    let held = null;
    try { held = window.sessionStorage.getItem(STORY_GAME_KEY); } catch (refused) { held = null; }
    const parts = (held || '').split(' ');
    if (parts.length === 2) kept = { key: parts[0], hash: parts[1] };
  }
  return kept && kept.hash === hash ? kept.key : null;
}

// The story's address, and the step its way back lands on: *what goes wrong*, the first beat
// the hero's door skipped.
const STORY_BACK_BEAT = 2;
let LANDING_BEAT = null;

function storyPath(key) {
  return '/?story=' + encodeURIComponent(key);
}

// A link back to the story, followed without loading the page again unless a new tab is asked.
function wireStoryBack(link) {
  link.addEventListener('click', event => {
    if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    LANDING_BEAT = STORY_BACK_BEAT;
    goPath(link.getAttribute('href'));
  });
}

// The gentlest computer player a game's page offers: the one it names for a first game, else
// the first of its ladder (weakest first), else the first it offers at all.
async function gentlestAgent(hash) {
  const game = await detailFor(hash);
  const offered = ((game && game.opponents) || []).filter(one => one.agent !== 'human');
  const first = game && game.firstGame && game.firstGame.gentle;
  if (first && offered.some(one => one.agent === first.agent)) return first.agent;
  const one = offered.find(o => o.store === 'ladder') || offered[0];
  return one ? one.agent : null;
}

async function playNow(door, hash, seat, agent) {
  if (door.getAttribute('aria-busy') === 'true') return;
  door.setAttribute('aria-busy', 'true');
  pending('Starting the game…');
  try {
    const against = agent || await gentlestAgent(hash);
    if (!against) { clearPending(); location.href = door.getAttribute('href'); return; }
    await startPractice(hash, against, seat);
  } catch (error) {
    fail(error, door);
  } finally {
    door.removeAttribute('aria-busy');
  }
}

// The template holds every story in the pool, each as one part (its hero and its journey,
// `landing.html_page`); the page keeps one and drops the rest before it is shown, so each
// part's ids stand once. The head's `story.js` drew it (LP6: `?story=<key>`, the tab's own on a
// reload, else one at random among the others); without that script, `?story=` or the first.
function chooseStory(fragment) {
  const parts = [...fragment.querySelectorAll('.storyPart')];
  if (!parts.length) return fragment;
  let wanted = typeof window.landingStory === 'function' ? window.landingStory() : null;
  if (!wanted) {
    try { wanted = new URLSearchParams(location.search).get('story'); } catch (err) { wanted = null; }
  }
  const kept = parts.find(part => part.dataset.story === wanted) || parts[0];
  parts.forEach(part => { if (part !== kept) part.remove(); });
  return fragment;
}

// A visitor to the site is offered *Sign in*, a member their own page in its place (Q723);
// a part this server has not got goes (`data-needs`).
function dropUnoffered(root) {
  const offered = { accounts: ACCOUNTS, round: ROUND,
    visitor: ACCOUNTS && !VIEWER, member: ACCOUNTS && !!VIEWER };
  root.querySelectorAll('[data-needs]').forEach(part => {
    if (!offered[part.dataset.needs]) part.remove();
  });
}

// The five shelves, as the band at the top of `/game` (LP3, `browse.js`'s `showShelves`),
// from the listing; the landing carries none since LP4.
function fillShelves(data) {
  if (data) takePieceArt(data);
  fillNewShelf(data);
  fillPicks(data);
  fillMembersShelf(data);
}

// LP3: *How the score works* (`#/score`), the score's long explanation as a view of its
// own: the server's markup (`landing.score_html`), cloned as the landing is. `target` is an
// id the landing's sections had (`#landing-score`, `#landingIngredients`), which an old link
// still names: the view scrolls to it, and opens the ingredients when they are what it names.
function showScore(target) {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  view.textContent = '';
  view.append(document.getElementById('scoreView').content.cloneNode(true));
  document.querySelectorAll('#sections a').forEach(link => link.removeAttribute('aria-current'));
  const named = target ? document.getElementById(target) : null;
  if (!named) return;
  if (named.tagName === 'DETAILS') named.open = true;
  named.scrollIntoView();
}

// Whether a bare fragment (`#landing-score`) is one of the ids the score's view keeps.
function scoreAnchor(part) {
  const template = document.getElementById('scoreView');
  return !!(part && template && template.content && template.content.getElementById(part));
}

// LP7: each word of the wall gains its door into the designer, `#/design?rule=<key>`, from the
// key the server put on it (`data-door`, `landing.door_link`'s spelling): the word becomes a
// chip of two links, its section of /rules and *start a game with this rule*.
const WALL_DOOR_SAID = 'start a game with this rule';

function wireWallDoors() {
  view.querySelectorAll('a.wallWord[data-door]').forEach(word => {
    const chip = document.createElement('span');
    chip.className = 'wallWord';
    const door = document.createElement('a');
    door.className = 'wallDoor';
    door.href = '#/design?rule=' + encodeURIComponent(word.dataset.door);
    door.title = WALL_DOOR_SAID;
    door.setAttribute('aria-label', WALL_DOOR_SAID + ': ' + word.textContent);
    door.textContent = '+';
    word.replaceWith(chip);
    word.className = 'wallRule';
    chip.append(word, door);
  });
}

// LP2: the wall of words. On a phone each family's words are one rail that scrolls sideways
// inside its row, faded at its right edge to say there is more; a rail scrolled to its end
// (or one that fits) is marked `atEnd`, and its last word stands unfaded.
function wireWallRails() {
  view.querySelectorAll('.wallRow dd').forEach(rail => {
    const mark = () => rail.classList.toggle(
      'atEnd', rail.scrollLeft + rail.clientWidth >= rail.scrollWidth - 2);
    rail.addEventListener('scroll', mark, { passive: true });
    mark();
  });
}

// Q788: *New in the library*, the newest games as the server named them (`web/launch.py`'s
// `new_cards`, newest first), each drawn with the games page's own thumbnail and score
// pill. `scoreTag` is the library card's: a game nobody has measured yet reads *not
// measured*, never a number, and a stale one is marked provisional. The shelf keeps one
// height whatever it holds; with no listing, or no new game offered, the section goes.
// Q826: which games are new is the server's, from the day each joined. When none joined
// inside the window the cards come marked not `fresh`, and the heading and the lead take
// their `data-older` words, so an older game is never called new.
function fillNewShelf(data) {
  const section = document.getElementById('landing-new');
  if (!section) return;
  const cards = (data && data.newGames) || [];
  const byHash = new Map(((data && data.games) || []).map(game => [game.hash, game]));
  const shown = cards.filter(card => byHash.has(card.hash));
  if (!shown.length) { section.remove(); return; }
  if (!shown.some(card => card.fresh)) {
    section.querySelectorAll('[data-older]').forEach(node => {
      node.textContent = node.dataset.older;
    });
  }
  // A day can add more games than the shelf holds, so its link opens the games page
  // ordered by the day each joined, newest first: the reader's order from then on, as if
  // they had picked *date added* from the menu. Without the page (a new tab), `/game`.
  const all = section.querySelector('[data-newest]');
  if (all) {
    all.onclick = event => {
      if (event.button || event.metaKey || event.ctrlKey || event.shiftKey) return;
      event.preventDefault();
      BROWSE.sort = 'added';
      BROWSE.descending = true;
      ORDER_SET = true;
      goPath(GAMES_PATH);
    };
  }
  const shelf = document.getElementById('newShelf');
  shelf.removeAttribute('aria-busy');
  const added = shelf.dataset.added || '{day}';
  shelf.innerHTML = shown.map(card => {
    const game = byHash.get(card.hash);
    return `<div class="newGame panel" data-open="/game/${esc(card.hash)}">
      ${thumbHtml(game) || '<div class="thumb"></div>'}
      <div class="newGameText">
        <a class="memberGameName" href="/game/${esc(card.hash)}">${esc(card.name)}</a>
        <p class="newGameSentence small" title="${esc(card.sentence)}">${esc(card.sentence)}</p>
        <div class="row newGamePills"><span class="tag newTag">${
          esc(added.replace('{day}', card.added))}</span>${scoreTag(game)}</div>
      </div>
    </div>`;
  }).join('');
  openWholeCards(shelf);
}

// First-visit study N6: a card opened only from its name, and a reader pressed its
// board. A shelf's card opens its game wherever it is pressed; a link inside it (the
// name, *Play*, a member's page) still goes where it says.
function openWholeCards(shelf) {
  shelf.onclick = event => {
    if (event.target.closest('a, button')) return;
    const card = event.target.closest('[data-open]');
    if (card) go(card.dataset.open);
  };
}

// Q788: the picks row, three panels the server filled: the day's game with its gentlest
// opponent a tap away (Q564, moved here from `/game`), the spotlight (*Intransitive*) and
// this month's game (Q603, moved here too). A panel the server sent nothing for goes, and
// the row with it when all three do. The words are the template's and the server's.
function fillPicks(data) {
  const section = document.getElementById('landing-picks');
  if (!section) return;
  const byHash = new Map(((data && data.games) || []).map(game => [game.hash, game]));
  const fill = (key, card, extra) => {
    const panel = document.getElementById(key);
    if (!panel) return;
    if (!card || !byHash.has(card.hash)) { panel.remove(); return; }
    const game = byHash.get(card.hash);
    const link = gamePath(card.hash);
    panel.removeAttribute('aria-busy');
    document.getElementById(key + 'Board').innerHTML = thumbHtml(game);
    const name = panel.querySelector('[data-name]');
    name.textContent = card.name;
    name.href = link;
    panel.querySelector('[data-sentence]').textContent = card.sentence || '';
    panel.querySelectorAll('[data-go], [data-find]').forEach(a => { a.href = link; });
    const why = panel.querySelector('[data-why]');
    if (why) why.textContent = card.why || '';
    extra(panel, game);
  };
  fill('landingToday', data && data.firstStep && data.firstStep.opponents ? data.firstStep : null,
    (panel) => wireTodaysGame(panel, data.firstStep));
  fill('landingSpotlight', data && data.spotlight, (panel, game) => {
    panel.querySelector('[data-score]').innerHTML = scoreTag(game);
  });
  fill('landingMonth', data && data.monthGame, (panel) => {
    panel.querySelector('[data-when]').textContent = ' · ' + data.monthGame.month;
    panel.querySelector('[data-how]').textContent = data.monthGame.how || '';
  });
  if (!section.querySelector('.pick')) section.remove();
}

// The day's game's two buttons (Q564, `docs/reference/opponent-line.md` §3.1): *Play the computer*
// seats the gentlest opponent the server named, as practice, and *A harder one* the next
// one up. The page picks no game and no opponent: it reads both off `firstStep`.
function wireTodaysGame(panel, step) {
  const gentle = step.opponents.gentle, harder = step.opponents.harder;
  panel.querySelector('[data-opponent]').textContent = gentle.name;
  startPracticeOn(panel.querySelector('[data-play]'), step.hash, gentle.agent);
  const more = panel.querySelector('[data-harder]');
  if (harder) {
    more.querySelector('span').textContent = harder.name;
    more.hidden = false;
    startPracticeOn(more, step.hash, harder.agent);
  }
}

// Q741: *Made by members*, the games members designed and published, as the server chose
// them (`site/showcase.py`: newest published first, at most four). The shelf keeps one
// height whatever it holds, so nothing below it moves when it fills. Before anybody has
// published, the template's invitation shows in the same space. With no listing at all
// the section goes, rather than say the shelf is empty when that is not known.
function fillMembersShelf(data) {
  const section = document.getElementById('landing-members');
  if (!section) return;
  const made = data && data.madeHere;
  if (!made) { section.remove(); return; }
  const shelf = document.getElementById('membersShelf');
  shelf.removeAttribute('aria-busy');
  // The cards' own pieces (Q762): the shelf draws each game as the games page's card does,
  // so it takes the listing's drawings first rather than finding whatever was taken last.
  takePieceArt(data);
  const games = made.games || [];
  if (!games.length) {
    document.getElementById('membersEmpty').hidden = false;
    return;
  }
  const byHash = new Map((data.games || []).map(game => [game.hash, game]));
  shelf.innerHTML = games.map(entry => {
    const game = byHash.get(entry.hash);
    // A member's page is for members; a visitor reads the name alone.
    const maker = `${markHtml(entry.mark, 16)}<span>${esc(entry.by)}</span>`;
    const by = VIEWER && entry.mark
      ? `<a href="/players/${encodeURIComponent(entry.by)}">${maker}</a>` : maker;
    return `<div class="memberGame panel" data-open="/game/${esc(entry.hash)}">
      ${game ? thumbHtml(game) : '<div class="thumb"></div>'}
      <div class="memberGameText">
        <a class="memberGameName" href="/game/${esc(entry.hash)}" title="${esc(entry.name)}">${esc(entry.name)}</a>
        <div class="memberGameBy small">by ${by}</div>
        <a class="button primary memberGamePlay" href="/game/${esc(entry.hash)}">Play</a>
      </div>
    </div>`;
  }).join('');
  openWholeCards(shelf);
  if (made.total > games.length) {
    document.getElementById('membersCount').textContent = String(made.total);
    document.getElementById('membersAll').hidden = false;
  }
}

// ------------------------------------------------------------------ the hero (LP4)
//
// `docs/plan-landing.md` §3.2: *Wouldn't it be cool if…* with its ending turning over every
// few seconds, the story's own first, each beside the library game that idea became, drawn
// from the listing's still with its marks (the seams, the walls, the walks, the goals, a
// hand). The words and the order are the server's (`landing._hero`); this finds each game
// in the listing by its name and draws it. It pauses on hover, on a press of a dot, with
// *pause*, and while the tab is hidden; under reduced motion it never turns by itself.
const HERO_TURN = 3800;

function heroStill(game) {
  const still = game && game.still;
  if (!still) return '';
  const icons = (still.icons || []).map(pair => (pair || []).map(icon =>
    icon && PIECE_ART[icon.stem]
      ? { href: PIECE_ART[icon.stem], rotate: icon.rotate, stem: icon.stem } : null));
  return boardPreview(Object.assign({}, still, { icons: icons, hash: game.hash }), null);
}

// The stories' games, asked once a page (`/api/stories`).
function storiesData() {
  if (!STORY.fetched) STORY.fetched = api('/api/stories').catch(() => null);
  return STORY.fetched;
}

async function heroFrame(state) {
  const data = await storiesData();
  const story = data && data.games && data.games[state.dataset.story];
  if (!story || !document.body.contains(state)) return;
  const frame = storyFrames(story.right).find(f => f.n === Number(state.dataset.at));
  if (!frame) return;
  drawStoryFrame(state.querySelector('[data-board]'), story.right, frame, null, data.pieceArt || {}, true);
}

function startHero(data) {
  const show = document.getElementById('heroShow');
  if (!show) return;
  const games = new Map(((data && data.games) || []).map(game => [game.name, game]));
  const clauses = [...document.querySelectorAll('#heroClauses .clause')];
  const states = [...show.querySelectorAll('.heroState')];
  const dots = [...show.querySelectorAll('.heroDot')];
  const pause = document.getElementById('heroPause');
  states.forEach(state => {
    const game = games.get(state.dataset.game);
    const link = state.querySelector('[data-name]');
    if (game && link) link.href = gamePath(game.hash);
    // A story's own game at the turn its hero names (`data-at`) is drawn from the story's
    // frames, as its steps draw it; a constructed story's game has no listing to draw from.
    if (state.dataset.at && state.dataset.story) heroFrame(state);
    else if (game) state.querySelector('[data-board]').innerHTML = heroStill(game);
  });
  let at = 0, timer = null, held = false;
  const turn = i => {
    const was = at;
    at = (i + states.length) % states.length;
    clauses.forEach((c, n) => {
      c.classList.toggle('on', n === at);
      c.classList.toggle('off', n === was && n !== at);
    });
    states.forEach((s, n) => s.classList.toggle('on', n === at));
    dots.forEach((d, n) => d.classList.toggle('on', n === at));
  };
  const onScreen = () => document.body.contains(show);
  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
    pause.textContent = 'play';
    pause.setAttribute('aria-pressed', 'true');
  };
  const play = () => {
    stop();
    pause.textContent = 'pause';
    pause.setAttribute('aria-pressed', 'false');
    timer = setInterval(() => {
      if (!onScreen()) { stop(); return; }
      if (!held && !document.hidden) turn(at + 1);
    }, HERO_TURN);
  };
  dots.forEach((dot, n) => dot.addEventListener('click', () => { stop(); turn(n); }));
  pause.addEventListener('click', () => (timer ? stop() : play()));
  show.addEventListener('mouseenter', () => { held = true; });
  show.addEventListener('mouseleave', () => { held = false; });
  if (STILL.matches || states.length < 2) { pause.hidden = true; return; }
  // LP9: a drawn story's hero holds on the story's own ending (a turning clause read as the
  // page changing under the reader, `landing-cold-read.md` §3 item 5); its dots and *play*
  // still turn it. With no story it turns until the reader first scrolls or taps.
  if (show.closest('.storyPart')) { stop(); return; }
  play();
  const settle = event => {
    if (event && event.target && event.target.closest && event.target.closest('#heroShow')) return;
    window.removeEventListener('scroll', settle);
    document.removeEventListener('pointerdown', settle, true);
    if (onScreen()) stop();
  };
  window.addEventListener('scroll', settle, { passive: true });
  document.addEventListener('pointerdown', settle, true);
}

// ------------------------------------------------------------------ the journey (LP4)
//
// §2.1-§2.2: the story's two games, *what goes wrong* and *what it changes*, played on the
// steps' boards with the rule drawn as it acts (`motion.js`), each verdict shown only once its
// game has ended. Every position is one the engine produced and every event (a move's two
// squares, a drop, a portal's two ends, a blast's area and what it cleared) is the server's,
// sent by `/api/stories` (`stories.replay`); this lays the positions out and draws the way
// between them, and works out nothing about the game (rule 3). At 1400 one board is pinned
// beside the steps and shows the step in view; on a phone (and under reduced motion, where
// nothing is pinned) steps 2 and 4 carry their own boards.

// A game's positions in order: the start, then each frame shown, with the squares it changed
// laid onto the one before.
function storyFrames(game) {
  let board = game.start.board.slice();
  let hands = game.start.hands || null;
  let counts = null;
  // Who holds each square, where the rule claims ground: the start's, then each frame's
  // changes (owner 0 where a claim went), as the server read them off the engine.
  let claims = game.start.claims ? new Map(game.start.claims) : null;
  if (claims) counts = game.start.counts || null;
  const out = [{ n: 0, board: board, hands: hands, counts: counts, claims: claims, item: null,
    key: true }];
  for (const item of game.frames) {
    board = board.slice();
    for (const [square, value] of item.set) board[square] = value;
    hands = item.hands || hands;
    counts = item.counts || counts;
    if (item.claims) {
      claims = new Map(claims || []);
      for (const [square, owner] of item.claims) {
        if (owner) claims.set(square, owner); else claims.delete(square);
      }
    }
    out.push({ n: item.n, move: item.move, board: board, hands: hands, counts: counts,
      claims: claims, item: item, key: !!item.key });
  }
  return out;
}

// The still payload for a frame: the game's own start, its pieces where the frame has them.
function storyPreview(game, frame, art) {
  const icons = (game.start.icons || []).map(pair => (pair || []).map(icon =>
    icon && art[icon.stem] ? { href: art[icon.stem], rotate: icon.rotate, stem: icon.stem } : null));
  return Object.assign({}, game.start, { board: frame.board, icons: icons, hash: game.hash,
    handsUsed: false, motion: true });
}

// Draw `frame` on `mount`, and the way from `prev` to it when there is motion to show and
// motion is wanted. Returns how long the motion takes.
function drawStoryFrame(mount, game, frame, prev, art, still) {
  // A plain turn moves within the story's pace; the turn the rule acts (a portal, a blast)
  // takes the time its motion needs (§2.2).
  const most = Math.max(90, game.pace * 0.9);
  const preview = storyPreview(game, frame, art);
  mount.innerHTML = boardPreview(preview, null);
  const svg = mount.querySelector('svg');
  if (!svg) return 0;
  const W = preview.width, H = preview.height, cell = 100 / Math.max(W, H);
  const view = flatView(W, H, FLIPPED, preview.lattice, boardCut(preview.absent, preview.wrap));
  const at = square => squareXY(square, cell, view);
  motionDrawer(svg, (value, x, y, size) => stillPieceHtml(preview, value, x, y, size));
  const item = frame.item;
  // The ground each side holds, under the pieces, in its claimer's colour (as the play board
  // paints it). With motion, a square this move claimed shows the paint it had before and
  // its new paint grows over it.
  const fresh = new Map();
  const painted = (frame.claims ? [...frame.claims] : []).concat(
    !still && prev && prev.claims && item && item.claims
      ? item.claims.filter(([sq]) => prev.claims.has(sq)).map(([sq]) => [sq, prev.claims.get(sq), true])
      : []);
  for (const [square, owner, was] of painted) {
    const cellGroup = svg.querySelector(`[data-square="${square}"]`);
    if (!cellGroup) continue;
    const [x, y] = at(square);
    const paint = document.createElementNS(SVGNS, 'rect');
    paint.setAttribute('x', x);
    paint.setAttribute('y', y);
    paint.setAttribute('width', cell);
    paint.setAttribute('height', cell);
    paint.setAttribute('fill', owner === 1 ? 'var(--claim0)' : 'var(--claim1)');
    paint.setAttribute('opacity', '0.45');
    paint.setAttribute('data-mark', 'claim');
    // The old paint goes under the new.
    cellGroup.insertBefore(paint, cellGroup.children[1] || null);
    if (!was) fresh.set(square, [paint, x, y]);
  }
  // The marks a still frame keeps, under the pieces: the move's two squares lit, a portal's
  // two ends, a blast's whole area, the squares a sweep cleared.
  const lit = !item ? [] : item.blast ? item.blast.area
    : [item.from, item.to].concat(item.portal || [], item.sweep || []).filter(sq => sq !== undefined);
  const burnt = new Set(item ? (item.blast ? item.blast.area : item.sweep || []) : []);
  for (const square of lit) {
    const cellGroup = svg.querySelector(`[data-square="${square}"]`);
    if (!cellGroup) continue;
    const [x, y] = at(square);
    const mark = document.createElementNS(SVGNS, 'rect');
    mark.setAttribute('x', x);
    mark.setAttribute('y', y);
    mark.setAttribute('width', cell);
    mark.setAttribute('height', cell);
    const hot = burnt.has(square);
    mark.setAttribute('fill', hot ? 'var(--fx-fire)' : 'var(--hl-last)');
    // A swept square is one square, so it takes a stronger tint than a blast's whole area.
    mark.setAttribute('opacity', item.blast ? '0.3' : hot ? '0.5' : '0.4');
    mark.setAttribute('data-mark', item.blast ? 'blast' : hot ? 'swept' : 'last');
    cellGroup.insertBefore(mark, cellGroup.children[1] || null);
  }
  if (still || !prev || !item) return 0;
  if (item.cut) {
    svg.animate([{ opacity: 0.35 }, { opacity: 1 }], { duration: 450, easing: 'ease-out' });
    return 450;
  }
  const before = prev.board;
  const node = square => svg.querySelector(`[data-piece="${square}"]`);
  if (item.drop !== undefined) {
    const piece = node(item.drop);
    if (!piece) return 0;
    const [x, y] = at(item.drop);
    return stoneDrop(piece, x, y, cell, 0, Math.max(most, Math.min(PACE.drop, game.pace * 2)));
  }
  if (item.from === undefined) return 0;
  const [fx, fy] = at(item.from), [tx, ty] = at(item.to);
  if (item.portal) {
    const [ex, ey] = at(item.portal[0]), [ox, oy] = at(item.portal[1]);
    const arriving = node(item.to);
    if (!arriving) return 0;
    const look = { ink: 'var(--pair-1)', mark: '', pair: '', end: '' };
    const trip = portalFlight(svg, arriving, before[item.from], {
      ex, ey, fx, fy, tx, ty, idx: ex - fx, idy: ey - fy, odx: tx - ox, ody: ty - oy,
      arc: false, wrapped: false, portal: item.portal[0], portalOut: item.portal[1],
      lookIn: look, lookOut: look }, cell, view);
    if (before[item.to]) capture(svg, before[item.to], tx, ty, cell, trip.blow);
    return trip.landed;
  }
  if (item.blast) {
    // The bomb's flight: the capturer travels to its target (a ghost, where the blast took
    // it too), then the fireball over the area and every piece it took thrown outward.
    const arriving = node(item.to);
    let flight;
    if (arriving) {
      flight = travel(arriving, fx - tx, fy - ty, false);
    } else {
      const bomb = ghostAt(svg, before[item.from], tx, ty, cell);
      flight = travel(bomb, fx - tx, fy - ty, false);
      bomb.animate([{ opacity: 1 }, { opacity: 0 }],
        { duration: PACE.blastLag * 2, delay: flight * PACE.impact, fill: 'forwards' });
    }
    const hit = flight * PACE.impact;
    let busy = blastFire(svg, item.blast.area, cell, view, hit);
    for (const square of item.blast.cleared.concat(arriving ? [] : [item.to])) {
      if (!before[square]) continue;
      const [x, y] = at(square);
      busy = Math.max(busy, burn(svg, before[square], x, y, cell, { at: hit, dx: 0, dy: 0 }, [tx, ty]));
    }
    return busy;
  }
  const moving = node(item.to);
  if (!moving) return 0;
  // A sweep is the rule acting: it takes the time its travel needs, not the story's pace.
  const flight = travel(moving, fx - tx, fy - ty, false, 0, item.sweep ? Infinity : most);
  // A sweep: each piece on its path is struck as the sweeper passes over it, in the order
  // the server listed them along the path.
  let swept = 0;
  for (const square of item.sweep || []) {
    if (!before[square]) continue;
    const [x, y] = at(square);
    const share = Math.hypot(x - fx, y - fy) / Math.max(1, Math.hypot(tx - fx, ty - fy));
    swept = Math.max(swept, capture(svg, before[square], x, y, cell,
      { at: flight * Math.min(1, share), dx: tx - fx, dy: ty - fy }));
  }
  if (swept) svg.appendChild(moving);
  // A square the move claimed is painted as the piece lands on it.
  let painting = 0;
  for (const [square, owner] of item.claims || []) {
    const paint = owner && fresh.get(square);
    if (paint) painting = Math.max(painting, claimPaint(paint[0], paint[1], paint[2], cell, flight * PACE.impact, most));
  }
  if (before[item.to]) {
    const ghost = before[item.to];
    const done = capture(svg, ghost, tx, ty, cell, { at: flight * PACE.impact, dx: tx - fx, dy: ty - fy });
    svg.appendChild(moving);
    return Math.max(flight, done, painting, swept);
  }
  return Math.max(flight, painting, swept);
}

// What the rule keeps count of, under a board: the pieces in hand, or the ground each side holds.
function storyTray(game, frame) {
  const start = game.start;
  const sides = start.sides || ['white', 'black'];
  const name = side => side.charAt(0).toUpperCase() + side.slice(1);
  if (frame.counts) {
    return frame.counts.map((count, p) => `<span class="handCount"><span class="claimKey p${p}" aria-hidden="true"></span>${esc(name(sides[p]))} ${count}</span>`)
      .join('') + '<span class="muted small">squares held</span>';
  }
  if (!frame.hands) return '';
  // `unlimited` is the compiled stock's word per type: true where a hand never runs out.
  const unlimited = new Set((start.unlimited || []).map((never, type) => (never === true ? type : -1))
    .filter(type => type >= 0));
  const types = [];
  (start.hands || []).forEach(hand => (hand || []).forEach((count, type) => {
    if ((count > 0 || unlimited.has(type)) && !types.includes(type)) types.push(type);
  }));
  return types.map(type => {
    const piece = (start.pieceNames || [])[type] || (start.pieceIds || [])[type] || '';
    return `<span class="muted small">${esc(piece)}s in hand</span>` + frame.hands.map((hand, p) =>
      `<span class="handCount">${esc(name(sides[p]))} ${unlimited.has(type) ? '∞' : (hand[type] || 0)}</span>`)
      .join('');
  }).join('');
}

const STORY = { active: 1, players: {}, data: null, fetched: null };

function pinShown() {
  const pin = document.querySelector('#journey .pin');
  return !!pin && pin.offsetParent !== null;
}

class StoryPlayer {
  // `hold`: the game opens on the turn its rule acts and holds there until the reader presses
  // play (LP9: step 2's moment, the door's turn 1 or the sweep's mate, was played past before
  // the reader looked).
  constructor(which, game, art, line, good, hold) {
    this.which = which;
    this.hold = !!hold;
    this.holding = false;
    this.game = game;
    this.art = art;
    this.line = line;
    this.good = good;
    const frames = storyFrames(game);
    // Under reduced motion only the key frames, stepped by press, opening on the moment the
    // rule acts; otherwise every frame, played.
    this.frames = STILL.matches ? frames.filter(f => f.key) : frames;
    const moment = this.frames.findIndex(f => f.n === game.moment);
    this.at = STILL.matches ? Math.max(0, moment) : 0;
    this.timer = null;
    this.played = false;
    this.ended = false;
    this.root = document.querySelector(`[data-ctl="${which}"]`).closest('.beat');
    this.board = this.root.querySelector(`[data-board="${which}"]`);
    this.words = this.root.querySelector(`[data-words="${which}"]`);
    this.tray = this.root.querySelector(`[data-tray="${which}"]`);
    this.verdict = this.root.querySelector(`[data-answer="${which}"]`);
    this.button = this.root.querySelector('[data-step="play"]');
    this.button.hidden = STILL.matches;
    this.root.querySelector('[data-step="back"]').addEventListener('click', () => {
      this.stop(); this.show(this.at - 1, true);
    });
    this.root.querySelector('[data-step="next"]').addEventListener('click', () => {
      this.stop(); this.show(this.at + 1, STILL.matches);
    });
    this.button.addEventListener('click', () => (this.timer ? this.stop()
      : this.play(this.at >= this.frames.length - 1 ? 0 : undefined)));
    this.show(this.at, true);
    if (STILL.matches) this.done();
  }
  pinned() { return STORY.active === (this.which === 'wrong' ? 2 : 4) && pinShown(); }
  show(i, still) {
    const prev = this.frames[this.at];
    this.at = Math.max(0, Math.min(this.frames.length - 1, i));
    const frame = this.frames[this.at];
    const from = prev === frame ? null : prev;
    let motion = 0;
    const boards = [this.board];
    if (this.pinned()) boards.push(document.getElementById('pinBoard'));
    for (const mount of boards) {
      if (mount.offsetParent === null) continue;
      motion = Math.max(motion, drawStoryFrame(mount, this.game, frame, from, this.art, still));
    }
    const words = frame.n ? `turn ${frame.n}: ${frame.move}${frame.item && frame.item.cut ? ' (later)' : ''}`
      : 'turn 0: the start';
    this.words.textContent = words;
    this.tray.innerHTML = storyTray(this.game, frame);
    if (this.pinned()) {
      document.getElementById('pinWords').textContent = words;
      document.getElementById('pinTray').innerHTML = storyTray(this.game, frame);
    }
    // The verdict waits for the last turn's motion (a blast, a journey) to be over.
    if (this.at === this.frames.length - 1) {
      if (motion) {
        const shown = this.at;
        setTimeout(() => { if (this.at === shown) this.done(); }, motion);
      } else {
        this.done();
      }
    }
    return motion;
  }
  done() {
    this.stop();
    this.ended = true;
    this.verdict.classList.remove('pending');
    if (this.pinned()) pinVerdict(this.good ? 'good' : 'bad', this.line);
  }
  stop() {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (!STILL.matches) this.button.textContent = 'play';
  }
  play(from) {
    this.stop();
    this.holding = false;
    if (from !== undefined) this.show(from, true);
    this.button.textContent = 'pause';
    const step = () => {
      this.timer = null;
      if (!document.body.contains(this.root)) return;
      if (document.hidden) { this.button.textContent = 'play'; return; }
      if (this.at >= this.frames.length - 1) { this.done(); return; }
      const motion = this.show(this.at + 1, false);
      const frame = this.frames[this.at];
      // The turn the rule acts (a portal, a blast) gets the time its motion needs and a
      // breath after it; any other turn takes the story's pace.
      const acts = frame.item && (frame.item.portal || frame.item.blast || frame.item.sweep);
      const wait = Math.max(this.game.pace, acts ? motion + 350 : motion)
        + (frame.item && frame.item.cut ? 600 : 0);
      if (this.at < this.frames.length - 1) {
        this.timer = setTimeout(step, wait);
        this.button.textContent = 'pause';
      }
    };
    this.timer = setTimeout(step, 500);
  }
  start() {
    if (STILL.matches || (this.played && this.at >= this.frames.length - 1)) {
      this.show(this.at, true);
      return;
    }
    if (this.played) {  // scrolled away mid-game: carry on, or stay on the moment
      if (this.holding) this.show(this.at, true); else this.play();
      return;
    }
    this.played = true;
    if (this.hold) { this.holdOnMoment(); return; }
    this.play(0);
  }
  // The turn before the moment, then the moment's own move played once, and a hold there.
  holdOnMoment() {
    const moment = Math.max(0, this.frames.findIndex(f => f.n === this.game.moment));
    this.holding = true;
    this.show(Math.max(0, moment - 1), true);
    if (!moment) return;
    setTimeout(() => {
      if (this.holding && !this.timer && document.body.contains(this.root)) this.show(moment, false);
    }, 500);
  }
}

function pinVerdict(kind, line) {
  const box = document.getElementById('pinVerdict');
  if (!box) return;
  box.className = 'pinVerdict mono small' + (kind ? ' shown ' + kind : '');
  box.textContent = kind ? line : '';
}

// The pinned board's six states (1400): the failed game's start, its game, the fixed game's
// start with the change shown in the tray, its game, then its first two moves.
function storyBeat(n) {
  STORY.active = n;
  document.querySelectorAll('#journey .beat').forEach(beat => {
    const on = Number(beat.dataset.beat) === n;
    beat.classList.toggle('active', on);
    if (on) beat.classList.add('seen');  // the card's edit stays made
  });
  const players = STORY.players;
  for (const [which, player] of Object.entries(players)) {
    if ((which === 'wrong' ? 2 : 4) !== n) player.stop();
  }
  if (!players.wrong || !pinShown()) {
    if (n === 2 && players.wrong) players.wrong.start();
    if (n === 4 && players.right) players.right.start();
    return;
  }
  if (n === 2 || n === 4) {
    const player = n === 2 ? players.wrong : players.right;
    pinVerdict(null, '');
    player.show(player.at, true);
    player.start();
    if (player.ended) pinVerdict(player.good ? 'good' : 'bad', player.line);
    return;
  }
  const right = players.right, wrong = players.wrong;
  const frames = n === 1 ? storyFrames(wrong.game) : storyFrames(right.game);
  const at = { 1: 0, 3: 0, 5: 1, 6: 2 }[n];
  const game = n === 1 ? wrong.game : right.game;
  const frame = frames[Math.min(at, frames.length - 1)];
  drawStoryFrame(document.getElementById('pinBoard'), game, frame, null, right.art, true);
  const words = { 1: `${STORY.data.failed}, as it starts`, 3: `${STORY.data.fixed}, as it starts`,
    5: frame.n ? `turn ${frame.n}: ${frame.move}` : '', 6: frame.n ? `turn ${frame.n}: ${frame.move}, by correspondence` : '' }[n];
  document.getElementById('pinWords').textContent = words;
  const tray = document.getElementById('pinTray');
  tray.innerHTML = storyTray(game, frame);
  pinVerdict(n >= 5 ? (right.good ? 'good' : 'bad') : null, right.line);
  if (n === 3) {
    // The change, in the tray: the failed game's hand struck, the fixed one's arriving.
    const was = storyTray(wrong.game, storyFrames(wrong.game)[0]);
    const now = storyTray(right.game, frame);
    if (was && now && was !== now && !STILL.matches) {
      tray.innerHTML = `<span class="swap"><del>${was}</del> <ins>${now}</ins></span>`;
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const swap = tray.querySelector('.swap');
        if (swap) swap.classList.add('go');
      }));
    }
  }
}

// The six qualities' bars: each one's length from its number, a CSSOM custom property.
function drawPairBars(root) {
  root.querySelectorAll('.pairBar [data-v]').forEach(bar => {
    const v = Number(bar.dataset.v);
    bar.style.setProperty('--v', String(v));
    bar.classList.toggle('low', v < 0.25);
  });
}

// Step 5: the fixed game's computer opponents as its own page offers them, and, on the site,
// the page's own sentences placing them on people's scale (`standing.peopleScale.said`, shown
// once enough people are rated there), read from the game's own payload, so every name and
// number is the page's.
function fillLadder(root) {
  const ladder = root.querySelector('[data-ladder]');
  if (!ladder) return;
  api('/api/game/' + encodeURIComponent(ladder.dataset.ladder)).then(game => {
    const offered = ((game && game.opponents) || []).filter(one => one.agent !== 'human');
    ladder.innerHTML = offered.map(one => `<span>${esc(one.name || one.agent)}</span>`).join('');
    const said = (((game && game.standing) || {}).peopleScale || {}).said || [];
    if (said.length) {
      ladder.insertAdjacentHTML('afterend',
        `<p class="small muted">${said.map(line => esc(line)).join(' ')}</p>`);
    }
  }).catch(() => {});
}

async function startJourney() {
  const journey = document.getElementById('journey');
  if (!journey) return;
  drawPairBars(journey);
  fillLadder(journey);
  const data = await storiesData();
  if (!document.body.contains(journey)) return;
  const story = data && data.games && data.games[journey.dataset.story];
  if (!story) return;
  STORY.data = story;
  const art = data.pieceArt || {};
  STORY.players = {
    wrong: new StoryPlayer('wrong', story.wrong, art, story.failedLine, false, true),
    right: new StoryPlayer('right', story.right, art, story.fixedLine, true),
  };
  const beats = journey.querySelectorAll('.beat');
  if (STILL.matches || !('IntersectionObserver' in window)) {
    beats.forEach(beat => beat.classList.add('seenAll'));
    return;
  }
  storyBeat(1);
  // The step nearest the middle of the screen sets the pinned board; on a phone a step's
  // own board plays its game when it comes into view.
  const watch = new IntersectionObserver(entries => {
    if (!document.body.contains(journey)) { watch.disconnect(); return; }
    for (const entry of entries) if (entry.isIntersecting) storyBeat(Number(entry.target.dataset.beat));
  }, { rootMargin: '-45% 0px -45% 0px' });
  beats.forEach(beat => watch.observe(beat));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) Object.values(STORY.players).forEach(player => player.stop());
  });
}
