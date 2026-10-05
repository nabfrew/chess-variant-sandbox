// The landing page (Q718): what everybody meets at `/`, signed in or not (Q723); the games
// are at `/game`.
//
// The words are the server's: `web/landing.py` makes the page once and the single page
// carries it as `<template id="landingPage">`, so this file draws no sentence of its own
// beyond a game's name and the server's one sentence about it. It clones the template into
// the view, hides what this server has not got (`data-needs`), and starts the opening board.
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
  // The board waits for the listing; the words above and below it do not.
  const data = await api('/api/library').catch(() => null);
  if (data) startLandingBoard(landingGames(data));
}

// The games the board shows, in order: today's game, then the starting games, each with
// the sentence the server wrote for it. Where the server sends no starting games (the
// local application) the first few games in the listing that have a board.
function landingGames(data) {
  const games = data.games || [];
  if (data.pieceArt) PIECE_ART = data.pieceArt;
  const byHash = new Map(games.map(game => [game.hash, game]));
  const named = [];
  const add = (hash, sentence) => {
    const game = byHash.get(hash);
    if (game && game.still && !named.some(entry => entry.game === game)) {
      named.push({ game: game, sentence: sentence || '' });
    }
  };
  if (data.firstStep) add(data.firstStep.hash, data.firstStep.sentence);
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
    mount.innerHTML = `<div class="landingSlide" style="width:${wide}%">${
      thumbHtml(entry.game)}</div>`;
    if (settle) {
      mount.querySelectorAll('svg image, svg circle, svg text').forEach((piece, n) => {
        piece.style.animationDelay = (n * LANDING_STEP) + 'ms';
        piece.classList.add('settle');
      });
    }
    caption.innerHTML = `<a href="#/game/${esc(entry.game.hash)}"><b>${
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
