// The landing page (Q718): what everybody meets at `/`, signed in or not (Q723); the games
// are at `/game`.
//
// The words are the server's: `web/landing.py` makes the page once and the single page
// carries it as `<template id="landingPage">`, so this file draws no sentence of its own
// beyond a game's name and the server's one sentence about it. It clones the template into
// the view, hides what this server has not got (`data-needs`), fills the newest games and
// the picks row from the listing (Q788: games, sentences and days are the server's, each
// score the library card's own pill), and starts the opening board.
//
// **The opening board** is the club room's quiet board: one game's opening position at a
// time, drawn by the still board the library's cards use (`thumbHtml`, so the wood and the
// pieces are the cards' own), its pieces set down one after another, held, then the next
// game. It shows the starting games, today's first. It plays no move, so the page decides
// nothing about any game (rule 3). Under `prefers-reduced-motion` it is one still board,
// with *another game* to step by hand; otherwise *pause* stops it, as anything that moves
// on its own for more than five seconds must offer. It stops when the view changes and
// waits while the tab is hidden, so a phone in a pocket spends nothing on it.

// How long one game's board is held, and how far apart its pieces are set down.
const LANDING_HOLD = 6500, LANDING_STEP = 45, LANDING_FADE = 500;

async function showLanding() {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  const template = document.getElementById('landingPage');
  view.textContent = '';
  view.append(template.content.cloneNode(true));
  // The landing page is in none of the header's sections, at `/` or at `#/about`.
  document.querySelectorAll('#sections a').forEach(link => link.removeAttribute('aria-current'));
  // A visitor to the site is offered *Sign in*, a member their own page in its place (Q723).
  const offered = { accounts: ACCOUNTS, round: ROUND,
    visitor: ACCOUNTS && !VIEWER, member: ACCOUNTS && !!VIEWER };
  view.querySelectorAll('[data-needs]').forEach(part => {
    if (!offered[part.dataset.needs]) part.remove();
  });
  wireWallRails();
  // The board waits for the listing; the words above and below it do not.
  const data = await api('/api/library').catch(() => null);
  if (data) takePieceArt(data);
  fillNewShelf(data);
  fillPicks(data);
  fillMembersShelf(data);
  if (data) startLandingBoard(landingGames(data));
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

// The games the board shows, in order: today's game, then the starting games, each with
// the sentence the server wrote for it. Where the server sends no starting games (the
// local application) the first few games in the listing that have a board.
function landingGames(data) {
  const games = data.games || [];
  takePieceArt(data);
  const byHash = new Map(games.map(game => [game.hash, game]));
  const named = [];
  const add = (hash, sentence, today) => {
    const game = byHash.get(hash);
    if (game && game.still && !named.some(entry => entry.game === game)) {
      named.push({ game: game, sentence: sentence || '', today: !!today });
    }
  };
  if (data.firstStep) add(data.firstStep.hash, data.firstStep.sentence, true);
  (data.launchSet || []).forEach(card => add(card.hash, card.sentence));
  if (!named.length) games.slice(0, 12).forEach(game => add(game.hash, ''));
  return named;
}

function startLandingBoard(named) {
  const mount = document.getElementById('landingBoard');
  if (!mount || !named.length) return;
  const caption = document.getElementById('landingCaption');
  const pause = document.getElementById('landingPause');
  const next = document.getElementById('landingNext');
  const moving = !STILL.matches && named.length > 1;
  let at = 0, paused = false, timer = null;
  const onScreen = () => document.body.contains(mount);

  const draw = (settle) => {
    const entry = named[at];
    const still = entry.game.still;
    // A board taller than it is wide is narrowed, so every game fits the same square.
    const wide = Math.min(100, 100 * still.width / Math.max(1, still.height));
    mount.innerHTML = `<div class="landingSlide ${styleClass(`width:${wide}%`)}">${
      thumbHtml(entry.game)}</div>`;
    if (settle) {
      mount.querySelectorAll('svg image, svg circle, svg text').forEach((piece, n) => {
        piece.style.animationDelay = (n * LANDING_STEP) + 'ms';
        piece.classList.add('settle');
      });
    }
    // The day's game says so (Q741), in the template's word, so the board's first game
    // is not read as one more of the starting games.
    const today = entry.today && caption.dataset.today
      ? `<span class="muted">${esc(caption.dataset.today)} &middot; </span>` : '';
    caption.innerHTML = `${today}<a href="/game/${esc(entry.game.hash)}"><b>${
      esc(entry.game.name)}</b></a>${entry.sentence ? ': ' + esc(entry.sentence) : ''}`;
  };

  const advance = () => {
    timer = null;
    if (!onScreen()) return;
    if (paused || document.hidden) { timer = setTimeout(advance, LANDING_HOLD); return; }
    const slide = mount.querySelector('.landingSlide');
    if (slide) slide.classList.add('leaving');
    setTimeout(() => {
      if (!onScreen()) return;
      at = (at + 1) % named.length;
      draw(true);
      timer = setTimeout(advance, LANDING_HOLD);
    }, LANDING_FADE);
  };

  draw(moving);
  if (named.length > 1) {
    next.hidden = false;
    next.onclick = () => {
      if (timer) clearTimeout(timer);
      at = (at + 1) % named.length;
      draw(moving && !paused);
      if (moving) timer = setTimeout(advance, LANDING_HOLD);
    };
  }
  if (moving) {
    pause.hidden = false;
    pause.onclick = () => {
      paused = !paused;
      pause.textContent = paused ? 'play' : 'pause';
      pause.setAttribute('aria-pressed', String(paused));
    };
    timer = setTimeout(advance, LANDING_HOLD);
  }
}
