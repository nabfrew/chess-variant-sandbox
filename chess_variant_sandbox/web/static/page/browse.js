// The library: what games there are, and finding the one you want.
//
// Forty-eight games is past the size a grid of cards answers a question about. The three
// things a person actually asks -- *which of these can I play right now*, *which ones are
// any good at the thing I care about*, and *what came out of what* -- are a filter, a sort
// and a grouping, and none of them existed.
//
// Two decisions worth stating:
//
// **The filters are the grid's own axes.** ``loop/grid.py`` already files every ruleset by
// eight things readable straight off its rules, each bin already carrying a sentence. So
// the facets are not a second vocabulary invented for the interface; they are the one the
// archive and the design loop already use, and a bin that appears here means exactly what
// it means there.
//
// **Sorting by a measurement names it in words.** The sort menu is the one place a
// component identifier would otherwise be unavoidable, and ``gradient_persistence`` is not
// a menu item. The titles come from ``metrics/glossary.py`` through ``metric_legend``, the
// same text the command line prints -- and when the list is ordered by one of them, every
// card shows that measurement, because a ranking whose criterion is invisible is a ranking
// you have to take on trust.

// ------------------------------------------------------------------ the library

async function showLibrary() {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  // A cold server takes twenty seconds to warm the listing, and until now the page said
  // nothing for all of it. The same spinner the game page shows, cleared when the
  // markup below replaces it.
  pending('Loading the library…');
  // Two requests rather than one: the library is what the games *are*, the survey is
  // what their scores rest on. They are asked together because the panel that offers to
  // fix a stale score has to be able to say how many there are before you press it.
  //
  // The survey is a route the site withholds, so it is asked for only where it is
  // served. `api` would refuse it anyway, but a refusal caught and thrown away is still
  // a 404 in the console on every landing.
  const [data, scoring, asked] = await Promise.all([
    api('/api/library'),
    offered('GET', '/api/scoring') ? api('/api/scoring').catch(() => null) : null,
    // Q571, Q572. A signed-in member's challenges and the games waiting on their move,
    // for the panel above the rest.
    ACCOUNTS && VIEWER ? api('/api/challenges').catch(() => null) : null,
  ]);
  SCORING = scoring;
  // Q931: the previews come apart from the library (`previews`), each with its level, so
  // the site files them under their own tab; nothing the server ranks or counts holds them.
  LIBRARY = data.games.concat(data.previews || []);
  takePieceArt(data);
  OPPONENTS = data.opponents;
  COVERAGE = data.coverage;
  METRICS = data.metrics;
  // Null on the local application, which has no store and so no stars, no human games
  // and no tag. Every control and every chip below is drawn only where this arrives.
  SITE = data.site || null;
  noteAdmin(SITE && SITE.admin);
  settleOrder();
  document.getElementById('tagline').textContent =
    // Archived games were invented here too -- keeping one must not make the count of
    // what this project has found go *down*. Committed is the only thing that isn't.
    `${data.games.length} games, ${
      data.games.filter(g => g.source !== 'committed').length} invented here${
      (data.previews || []).length ? `, and ${data.previews.length} preview${
        data.previews.length === 1 ? '' : 's'}` : ''}`;

  // Q788 had this page open on the search and the filters, the shelves being the landing's.
  // LP3 (`docs/plan-landing.md` §4.2, the user's decision of 2026-10-24) moves them here:
  // the landing becomes a story, and `/game` is where a visitor goes to find a game, so
  // the five shelves are one band at its top, above the search (`showShelves`). What waits
  // on a signed-in member (`forYou`) is theirs, not a highlight, and stays above it.
  view.innerHTML = `
    <div id="forYou">${forYouHtml(asked)}</div>
    <div id="shelvesMount"></div>
    <div class="row u-jc-space-between u-mb-12">
      <div class="muted small">Every game here is playable in the browser. Pick one, or
        <b>build your own</b>.</div>
      <div class="row">
        ${viewSwitchHtml('list')}
        <button class="primary" id="newGame">build a game</button>
      </div>
    </div>
    <div id="everything">
      <div id="scoring">${scoringHtml(SCORING)}</div>
      ${provisionalScoresHtml(data.provisionalScores)}
      <div id="tabs">${tabsHtml()}</div>
      ${controlsHtml()}
      <div id="results"></div>
    </div>`;

  showShelves(data);
  document.getElementById('newGame').onclick = () => go('#/design');
  // The same two buttons the map draws, so the way to the other view is in one place on
  // both. `graph.py` owns them.
  wireViewSwitch();
  wireForYou();
  wireTabs();
  wireControls();
  wireScoring();
  paintResults();
  if (SCORING && SCORING.sweep && SCORING.sweep.state === 'running') pollSweep();
}

// LP3: the band of shelves (`landing.shelves_html`, `<template id="shelvesBand">`), filled
// from the listing this page already holds, by the landing's own `fillShelves`. A shelf the
// server sent nothing for goes, as on the landing, and the band with it when all do.
function showShelves(data) {
  const mount = document.getElementById('shelvesMount');
  const template = document.getElementById('shelvesBand');
  if (!mount || !template) return;
  mount.append(template.content.cloneNode(true));
  dropUnoffered(mount);
  fillShelves(data);
  // On this page the members' *all of them* link would open the page it is on.
  const all = document.getElementById('membersAll');
  if (all) all.remove();
  // Before anybody has published, the landing invites the first; a band of picks has
  // nothing to show, so the empty shelf goes.
  const empty = document.getElementById('membersEmpty');
  if (empty && !empty.hidden) document.getElementById('landing-members').remove();
  if (!mount.querySelector('.landingPart')) mount.remove();
}

// The leaderboard's header (EV3, `docs/archive/plan-evolving-score.md` §5.2): how many of the
// listing's numbers are provisional while they are measured again. The server counts and
// words it; null when none is, and then the page says nothing at all.
function provisionalScoresHtml(found) {
  if (!found || !found.sentence) return '';
  return `<div id="provisionalScores" class="muted small u-mb-8">${
    esc(found.sentence)}</div>`;
}

// ------------------------------------------- what waits on you
//
// Q571 and Q572, `docs/archive/plan-return.md` §1.5 and s3 of the mock: above everything else on
// the landing page, a signed-in member's **Games waiting on your move** (the server's
// `waiting`: their running games between people where the engine's side to move is their
// seat, in `correspondence.started_order`) and **A challenge for you**, each with its one
// button. **Absent when there are none**: no empty heading, no nought, nothing in the
// header or the tab's title (test 3). Games waiting on the other person are on the
// person's own page. The person comes and looks; nothing tells them to. When the last
// move landed is a fact, and so is the time left to move, in the server's words; the
// page works out no time of its own and nothing here ticks.

//
// Q733: a game where this browser holds a move picked and never sent (`heldWaiting`,
// kept by the board) says so on its line, and its button says what is left to do. Only
// this browser knows it, so the line says *on this device*; another device shows the
// game as waiting, which it is.
function waitingHtml(asked) {
  const waiting = (asked && asked.waiting) || [];
  if (!waiting.length) return '';
  return `<h3 class="u-mt-0">Games waiting on your move</h3>
    ${waiting.map(entry => {
      const held = heldWaiting(entry.id, entry.plies);
      return `<div class="row u-mt-6">
      <span><a href="#/play/${esc(entry.id)}">${esc(entry.name)}</a> against
        <b>${markHtml(entry.opponentMark, 18)}${esc(entry.opponent)}</b>
        <span class="muted small">&middot; ${entry.rated ? 'rated' : 'practice'} &middot; ${
          entry.plies ? esc(entry.opponent) + ' moved ' : 'started '}${
          esc(corrAgoWords(entry.moved))} &middot; ${esc(entry.toMove)}</span>${held
          ? `<div class="held-note" data-held-waiting="${esc(entry.id)}"><b>Not sent yet:</b>
            you picked a move on this device and did not send it. ${esc(entry.opponent)}
            has not seen it.</div>` : ''}</span>
      <button class="primary" data-play="${esc(entry.id)}">${
        held ? 'Send your move' : 'Your move'}</button>
    </div>`;
    }).join('')}`;
}

function forYouHtml(asked) {
  const waiting = waitingHtml(asked);
  const challenged = (asked && asked.forYou) || [];
  if (!waiting && !challenged.length) return '';
  return `<div class="panel u-mb-14">
    ${waiting}
    ${challenged.length ? `<h3 class="${waiting ? 'u-mt-14' : 'u-mt-0'}"
      >A challenge for you</h3>` : ''}
    ${challenged.map(entry => `<div class="row u-mt-6">
      <span><b>${markHtml(entry.fromMark, 18)}${esc(entry.from)}</b> ${entry.rematch ? 'offers a rematch at'
        : 'challenged you to'} <a href="/game/${esc(entry.game)}">${esc(entry.name)}</a>
        <span class="muted small">&middot; ${esc(entry.seat)} &middot; ${
          entry.rated ? 'rated' : 'practice'}</span></span>
      <button class="primary" data-accept="${esc(entry.id)}">accept</button>
      <button data-decline="${esc(entry.id)}">decline</button>
    </div>`).join('')}
  </div>`;
}

function wireForYou() {
  const where = document.getElementById('forYou');
  if (!where) return;
  where.querySelectorAll('[data-play]').forEach(button => {
    button.onclick = () => go('#/play/' + button.dataset.play);
  });
  where.querySelectorAll('[data-accept]').forEach(button => {
    button.onclick = () => {
      button.disabled = true;
      pending('Starting the game…');
      api('/api/challenge/' + button.dataset.accept + '/accept', {})
        .then(made => go('#/play/' + made.id))
        .catch(error => { button.disabled = false; fail(error, button); });
    };
  });
  where.querySelectorAll('[data-decline]').forEach(button => {
    button.onclick = () => api('/api/challenge/' + button.dataset.decline + '/decline', {})
      .then(fresh => { where.innerHTML = forYouHtml(fresh); wireForYou(); })
      .catch(error => fail(error, button));
  });
}

// A button that seats you first against `agent` on the game `hash`, as a practice game
// (on the site; the local application has no other kind). The landing page's day's game
// (Q564's first step, there since Q788) wires its two buttons through this, so the landing
// page's own script starts no game.
function startPracticeOn(button, hash, agent) {
  if (!button) return;
  button.onclick = () => {
    button.disabled = true;
    startPractice(hash, agent, 0).catch(error => { button.disabled = false; fail(error, button); });
  };
}

// A practice game of `hash` against `agent`, you in seat `seat` (0 first, 1 second), opened
// once the server has started it. LP9: the landing's story doors start theirs through this
// too, in the seat their story is about.
function startPractice(hash, agent, seat) {
  pending('Starting the game…');
  const body = { hash: hash, seats: seat === 1 ? [agent, 'human'] : ['human', agent] };
  if (ACCOUNTS) body.mode = 'sandbox';
  return api('/api/session', body).then(open_);
}

// ---------------------------------------------------------------------------------------

// Which order the list opens in, decided once, on the first listing that arrives.
//
// The site opens on the project's own judgement, best first (the user, 2026-09-17:
// *"Decision: A - by score"*). Alphabetical said nothing about which of seventy games
// is worth an evening, and *almost chess, antichess, atomic, bastion* is what a visitor
// met. A game whose score is out of date has no number to sort on -- the number is
// withheld rather than shown stale, which is the corpus rule -- so it sorts last beside
// the unmeasured, and the deliberately dull anchors sit at the bottom under their own
// low numbers, which is honest.
//
// The local application keeps the alphabet: it is a working tool over a checkout, and
// the score panel above the list is already about which scores are current.
//
// First-visit study N5: a list opened *by score* over games none of which has a score is
// ordered by nothing, under a word that says otherwise. So the score is the opening order
// only where some game in the list has one to sort on; otherwise the list opens by name.
function settleOrder() {
  if (ORDER_SET) return;
  ORDER_SET = true;
  if (!ACCOUNTS) return;
  BROWSE.sort = openingSort('index', inTab());
  BROWSE.descending = BROWSE.sort !== 'name';
}

// The order a list opens in: `wanted`, unless no game in `games` has a value for it.
function openingSort(wanted, games) {
  return games.some(game => sortValue(game, wanted) !== null) ? wanted : 'name';
}

// ---------------------------------------------------------- narrowing and ordering

// What a card can be ordered by that is not a measurement. Everything else on the menu
// is generated from the glossary, so adding a component to the weights adds it here.
const PLAIN_SORTS = [
  ['name', 'name'],
  ['index', 'score'],
  // Q826: the day the game joined the library (a member's game: the day it was published),
  // `YYYY-MM-DD` on the card as `added`, so it sorts as text. Newest first when picked.
  ['added', 'date added'],
  ['squares', 'board size'],
  ['pieceTypes', 'number of piece types'],
  ['ruleMass', 'complexity'],
  ['descent', 'how many versions it has been through'],
];

// The five keys a player has a reason to reach for, in this order, above every group in
// the menu. The other twenty-two entries are the scoring formula's table of contents,
// which is a way in for somebody who already knows what a component is. Two of the five
// are the store's own columns and arrive with the listing, so they are named by the key
// a card carries and take the server's label; where there is no store the tier is three.
const FIRST_SORTS = ['name', 'index', 'added', 'stars', 'humanGames'];

const GROUPINGS = [
  ['none', 'nothing'],
  ['family', 'what it came from'],
  ['cell', 'kind of game'],
  ['source', 'where it is kept'],
  ['run', 'the run that made it'],
];

// Two of the five ask where a file is kept and which design run wrote it, which are
// questions about this repository rather than about a game. The site does not offer
// them, the same reason the card does not say "yardstick".
const groupings = () =>
  GROUPINGS.filter(([key]) => !(ACCOUNTS && (key === 'source' || key === 'run')));

// What the store lets a list be ordered by, in the server's own words, or nothing on the
// local application. Each entry names the card's key, so the ordering below needs no
// special case for them.
const siteSorts = () => ((SITE && SITE.sorts) || []).map(s => [s.key, s.label]);

// The value a game sorts on, or null where it has none. Null always sorts last, in both
// directions: "unscored" is not "scored zero", and a descending sort that put every
// unmeasured game at the top would be answering a different question.
function sortValue(game, key) {
  if (key === 'name') return game.name.toLowerCase();
  if (key.startsWith('family:')) return (game.families || {})[key.slice(7)] ?? null;
  if (key.startsWith('component:')) return (game.components || {})[key.slice(10)] ?? null;
  const value = game[key];
  return (value === undefined || value === '') ? null : value;
}

// ---------------------------------------------------------------------- the tabs
//
// Q503b, §3.2 of `docs/archive/plan-game-pipeline.md`. The site's listing opens on its Library
// and Baseline games, then the shortlisted, and a second tab lists the games on trial.
// The server sends the tabs with the levels each lists (`stages.TABS`); a card is in a
// tab when its `level` is one of them, and the filters apply within the tab. A checkout
// has no trial and is sent no tabs, so it lists everything as before.
function siteTabs() {
  return (SITE && SITE.tabs) || [];
}

function currentTab() {
  const tabs = siteTabs();
  return tabs.find(tab => tab.key === BROWSE.tab) || tabs[0] || null;
}

// The games in the current tab, before any filter: every game where there are no tabs.
function inTab() {
  const tab = currentTab();
  return tab ? LIBRARY.filter(game => tab.levels.includes(game.level)) : LIBRARY;
}

function tabsHtml() {
  const tabs = siteTabs();
  if (tabs.length < 2) return '';
  const now = currentTab();
  return `<div class="row u-gap-6 u-mb-10" role="tablist">${
    tabs.map(tab => `<button role="tab" data-tab="${esc(tab.key)}"
      aria-selected="${tab === now}" class="${tab === now ? 'primary' : ''}"
      >${esc(tab.label)} (${Number(tab.count) || 0})</button>`).join('')}</div>`;
}

// A tab opens in its own order: the trial tab newest first, the library by score, best
// first, as the site's listing always opened.
function wireTabs() {
  document.querySelectorAll('[data-tab]').forEach(button => {
    button.onclick = () => {
      BROWSE.tab = button.dataset.tab;
      const tab = currentTab();
      BROWSE.sort = openingSort((tab && tab.sort) || 'index', inTab());
      BROWSE.descending = BROWSE.sort !== 'name';
      const box = document.getElementById('tabs');
      if (box) box.innerHTML = tabsHtml();
      wireTabs();
      repaintControls();
    };
  });
}

function narrowed() {
  const text = BROWSE.text.trim().toLowerCase();
  return inTab().filter(game => {
    if (text) {
      const hay = `${game.name} ${game.description} ${game.hash} ${(game.cellBins || []).join(' ')}`;
      if (!hay.toLowerCase().includes(text)) return false;
    }
    if (BROWSE.stage && game.source !== BROWSE.stage) return false;
    if (BROWSE.playable === 'yes' && game.playable === false) return false;
    if (BROWSE.playable === 'no' && game.playable !== false) return false;
    // Whether the rules translate, which is what a card can honestly carry -- see
    // library._fairy. "no" is a refusal with reasons; "yes" is a candidate, and the
    // game's own page is where an installed engine either plays it or does not.
    if (BROWSE.fairy === 'yes' && game.fairyTranslates !== true) return false;
    if (BROWSE.fairy === 'no' && game.fairyTranslates !== false) return false;
    const stale = !!(game.staleness && !game.staleness.current);
    // *Scored* is a current number; a provisional one (EV2) is out of date, under `stale`.
    if (BROWSE.scored === 'yes' && (game.index === null || stale)) return false;
    if (BROWSE.scored === 'no' && (game.index !== null || stale)) return false;
    if (BROWSE.scored === 'stale' && !stale) return false;
    if (BROWSE.scored === 'viable' && game.viable !== true) return false;
    // The store's tag. Undefined on the local application, and null on a game whose
    // computer players have never been rated -- and "unrated" is not "unbeaten", so an
    // unrated game is out of both lists rather than in the first.
    if (BROWSE.tag === 'unbeaten' && game.unbeaten !== true) return false;
    if (BROWSE.tag === 'beaten' && game.unbeaten !== false) return false;
    if (BROWSE.tag === 'unrated' && game.rated !== false) return false;
    // Q972: the server's count of the proven positions the game's page offers
    // (`proven.offered`), undefined on the local application, which offers none.
    if (BROWSE.proven === 'yes' && !(game.provenPositions > 0)) return false;
    for (const [axis, wanted] of Object.entries(BROWSE.bins)) {
      if (!wanted.length) continue;
      if (!wanted.includes(binOf(game, axis))) return false;
    }
    return true;
  });
}

// Which bin a game has on one axis. The card carries the bins in the grid's own order,
// so the axis's position in the legend is the position in the list -- the same join the
// coverage table makes, and the reason neither has to carry the axis name per game.
function binOf(game, axis) {
  const axes = (COVERAGE && COVERAGE.axes) || [];
  const index = axes.findIndex(a => a.axis === axis);
  return index < 0 ? null : (game.cellBins || [])[index];
}

function ordered(games) {
  const key = BROWSE.sort;
  const sign = BROWSE.descending ? -1 : 1;
  return games.slice().sort((a, b) => {
    const left = sortValue(a, key), right = sortValue(b, key);
    if (left === null && right === null) return a.name.localeCompare(b.name);
    if (left === null) return 1;
    if (right === null) return -1;
    if (left === right) return a.name.localeCompare(b.name);
    return (left > right ? 1 : -1) * sign;
  });
}

// The groups, in the order they should appear, each already ordered inside itself.
// Grouping by lineage is the one that earns this: a design run makes a family of near
// relatives, and forty cards in alphabetical order is exactly the arrangement that hides
// which four of them are revisions of the same idea.
function grouped(games) {
  if (BROWSE.group === 'none') return [{ title: '', note: '', games }];
  const buckets = new Map();
  for (const game of games) {
    const [key, title, note] = groupOf(game);
    if (!buckets.has(key)) buckets.set(key, { key, title, note, games: [] });
    buckets.get(key).games.push(game);
  }
  const found = [...buckets.values()];
  if (BROWSE.group === 'family') {
    // A family whose oldest member is one game is not a family, and forty singleton
    // headings would bury the four groups that are the point of this view. So the
    // singletons are gathered at the end, under what they are: games nothing here was
    // revised from.
    const alone = found.filter(bucket => bucket.games.length === 1);
    const families = found.filter(bucket => bucket.games.length > 1);
    families.sort((a, b) => b.games.length - a.games.length || a.title.localeCompare(b.title));
    if (!alone.length) return families;
    return families.concat([{
      key: '', title: 'on their own',
      note: 'nothing here descends from these, and they descend from nothing here',
      games: alone.map(bucket => bucket.games[0])
        .sort((a, b) => a.name.localeCompare(b.name)),
    }]);
  }
  return found.sort((a, b) =>
    b.games.length - a.games.length || a.title.localeCompare(b.title));
}

function groupOf(game) {
  if (BROWSE.group === 'family') {
    const family = game.family || { hash: game.hash, name: game.name };
    return [family.hash, family.name || nameOf(family.hash) || family.hash, ''];
  }
  if (BROWSE.group === 'cell') {
    const bins = game.cellBins || [];
    return [game.cell, bins.slice(0, 4).join(', '), bins.slice(4).join(', ')];
  }
  if (BROWSE.group === 'run') {
    return [game.run || '-', game.run || 'not from a design run',
      game.theme ? 'asked to explore: ' + game.theme : ''];
  }
  const where = { committed: 'part of the yardstick', archived: 'kept in the repository',
                  designed: 'built by hand here', evolved: 'from a design run' };
  return [game.source, where[game.source] || game.source, ''];
}

// A measurement's name and its *what* line on the sort menu and the card, in the site's
// words where this is the site and the glossary has some (`library.metric_legend`, B3).
function legendTitle(entry, key) {
  return (ACCOUNTS && entry.site && entry.site.title) || entry.title || key;
}

function legendSays(entry) {
  return (ACCOUNTS && entry.site && entry.site.what) || entry.means || '';
}

function nameOf(hash) {
  const found = LIBRARY.find(g => g.hash === hash);
  return found ? found.name : '';
}

// ------------------------------------------------------------------- the controls

function controlsHtml() {
  const metrics = METRICS || { families: {}, components: {} };
  const option = (value, label, chosen, title) =>
    `<option value="${esc(value)}"${value === chosen ? ' selected' : ''}${
      title ? ` title="${esc(title)}"` : ''}>${esc(label)}</option>`;

  const families = Object.entries(metrics.families || {})
    .sort((a, b) => b[1].weight - a[1].weight)
    .map(([key, entry]) => option('family:' + key, entry.title, BROWSE.sort, entry.means));
  const components = Object.entries(metrics.components || {})
    .sort((a, b) => b[1].weight - a[1].weight)
    .map(([key, entry]) => option('component:' + key, legendTitle(entry, key), BROWSE.sort,
      legendSays(entry)));

  const axes = (COVERAGE && COVERAGE.axes) || [];
  const facets = axes.map(axis => `
    <div class="facet">
      <b title="${esc(axis.asks)}">${esc(axis.axis)}</b>
      <span class="muted small">${esc(axis.asks)}</span>
      <div class="chips">${axis.bins.map(bin => {
        const on = (BROWSE.bins[axis.axis] || []).includes(bin.bin);
        return `<button class="chip${on ? ' on' : ''}" data-axis="${esc(axis.axis)}"
          data-bin="${esc(bin.bin)}" title="${esc(bin.means)}">${esc(bin.bin)}</button>`;
      }).join('')}</div>
    </div>`).join('');

  const chosen = BROWSE.bins;
  const active = Object.values(chosen).reduce((n, list) => n + list.length, 0)
    + (BROWSE.stage ? 1 : 0) + (BROWSE.scored ? 1 : 0) + (BROWSE.playable ? 1 : 0)
    + (BROWSE.fairy ? 1 : 0) + (BROWSE.tag ? 1 : 0) + (BROWSE.proven ? 1 : 0);

  // Every plain ordering by key, the store's beside the file's, so the two tiers below
  // are a partition of one list rather than two lists that might disagree.
  const named = new Map([...PLAIN_SORTS, ...siteSorts()]);
  const first = FIRST_SORTS.filter(key => named.has(key));
  const rest = [...named.keys()].filter(key => !first.includes(key));
  const sortOption = key => option(key, named.get(key), BROWSE.sort);

  // A filter's (i) sits inside its label, so at phone width the two wrap together rather
  // than leaving one filter's (i) at the head of the next filter's line (Q972).

  return `<div class="controls panel">
    <div class="line">
      <input type="search" id="q" placeholder="search names, descriptions and rules"
        value="${esc(BROWSE.text)}">
      <label class="pick">order by
        <select id="sort">
          ${first.map(sortOption).join('')}
          <optgroup label="more about the game itself">${
            rest.map(sortOption).join('')}</optgroup>
          <optgroup label="by one part of the score">${
            families.concat(components).join('')}</optgroup>
        </select>
      </label>
      <button id="dir" title="reverse the order">${dirWords()}</button>
      <label class="pick">group by
        <select id="group">${groupings().map(([key, label]) =>
          option(key, label, BROWSE.group)).join('')}</select>
      </label>
    </div>
    <details${active ? ' open' : ''}>
      <summary class="muted small">narrow it down${active ? ` &mdash; ${active} filter${
        active === 1 ? '' : 's'} on` : ''}</summary>
      <div class="u-mt-8">
        <div class="line u-mb-6">
          ${ACCOUNTS ? '' : `<label class="pick">kept
            <select id="stage">
              ${option('', 'anywhere', BROWSE.stage)}
              ${option('committed', 'part of the yardstick', BROWSE.stage)}
              ${option('archived', 'in the repository', BROWSE.stage)}
              ${option('designed', 'built by hand', BROWSE.stage)}
              ${option('evolved', 'from a design run', BROWSE.stage)}
            </select></label>`}
          <label class="pick">score
            <select id="scored">
              ${option('', 'any', BROWSE.scored)}
              ${option('yes', 'has one', BROWSE.scored)}
              ${option('viable', 'has one, not ruled out', BROWSE.scored)}
              ${option('stale', 'out of date', BROWSE.scored)}
              ${option('no', 'never measured', BROWSE.scored)}
            </select></label>
          <label class="pick">playable
            <select id="playable">
              ${option('', 'any', BROWSE.playable)}
              ${option('yes', 'works as a game', BROWSE.playable)}
              ${option('no', 'ruled out', BROWSE.playable)}
            </select>${info('gated')}</label>
          ${ACCOUNTS ? '' : `<label class="pick">Fairy-Stockfish
            <select id="fairy">
              ${option('', 'any', BROWSE.fairy)}
              ${option('yes', 'the rules translate', BROWSE.fairy)}
              ${option('no', 'no counterpart', BROWSE.fairy)}
            </select>${info('fairy')}</label>`}
          ${SITE ? `<label class="pick">Unbeaten
            <select id="tag">
              ${option('', 'any', BROWSE.tag)}
              ${option('unbeaten', 'Unbeaten', BROWSE.tag)}
              ${option('beaten', 'beaten', BROWSE.tag)}
              ${option('unrated', 'unrated (no computer players yet)', BROWSE.tag)}
            </select>${info('unbeaten')}</label>` : ''}
          ${SITE ? `<label class="pick">proven positions
            <select id="proven">
              ${option('', 'any', BROWSE.proven)}
              ${option('yes', 'has some to solve', BROWSE.proven)}
            </select>${info('provenPositions')}</label>` : ''}
          <button id="clear">clear</button>
        </div>
        <div class="muted small">These are eight things read straight off the
          rules${info('kind')}. Pick any number; a row with nothing picked means "any".</div>
        ${facets}
      </div>
    </details>
  </div>`;
}

// The order button's words. A day is not high or low: *date added* reads newest or oldest
// first (Q826).
function dirWords() {
  if (BROWSE.sort === 'added') return BROWSE.descending ? 'newest first' : 'oldest first';
  return BROWSE.descending ? 'high to low' : 'low to high';
}

function wireControls() {
  const box = view.querySelector('.controls');
  if (!box) return;
  const q = box.querySelector('#q');
  // Repainted on every keystroke, and the results only: this is forty rows of client-side
  // filtering, so debouncing it would add lag to hide work that costs nothing.
  q.oninput = () => { BROWSE.text = q.value; paintResults(); };
  box.querySelector('#sort').onchange = event => {
    BROWSE.sort = event.target.value;
    // A measurement asked for by name is nearly always wanted best-first, and a name is
    // wanted A-Z. Guessing once beats making everybody's first act a second click.
    BROWSE.descending = BROWSE.sort !== 'name';
    repaintControls();
  };
  box.querySelector('#dir').onclick = () => {
    BROWSE.descending = !BROWSE.descending;
    repaintControls();
  };
  box.querySelector('#group').onchange = event => {
    BROWSE.group = event.target.value; paintResults();
  };
  ['stage', 'scored', 'playable', 'fairy', 'tag', 'proven'].forEach(key => {
    const control = box.querySelector('#' + key);
    if (!control) return;
    control.onchange = event => { BROWSE[key] = event.target.value; repaintControls(); };
  });
  box.querySelector('#clear').onclick = () => {
    BROWSE.bins = {}; BROWSE.stage = ''; BROWSE.scored = ''; BROWSE.playable = '';
    BROWSE.fairy = ''; BROWSE.tag = ''; BROWSE.proven = '';
    repaintControls();
  };
  box.querySelectorAll('[data-axis]').forEach(chip => {
    chip.onclick = () => {
      const axis = chip.dataset.axis, bin = chip.dataset.bin;
      const chosen = BROWSE.bins[axis] || [];
      BROWSE.bins[axis] = chosen.includes(bin)
        ? chosen.filter(b => b !== bin) : chosen.concat([bin]);
      repaintControls();
    };
  });
}

// The controls redraw themselves because a chip's own state is in them; the cursor is
// put back in the search box only when it was there, so clicking a chip does not steal
// focus from nothing.
//
// The filter drawer is left exactly as open or shut as the reader had it. `controlsHtml`
// opens it when a filter is on, which is right on a *first* draw and wrong on every
// repaint: this redraws on every chip, so anybody who shut the drawer while a filter was
// still on would have it spring back open under their cursor.
function repaintControls() {
  const box = view.querySelector('.controls');
  const focused = document.activeElement && document.activeElement.id === 'q';
  const drawer = box && box.querySelector('details');
  const open = drawer ? drawer.open : null;
  if (box) box.outerHTML = controlsHtml();
  const now = view.querySelector('.controls details');
  if (now && open !== null) now.open = open;
  wireControls();
  if (focused) {
    const q = view.querySelector('#q');
    if (q) { q.focus(); q.setSelectionRange(q.value.length, q.value.length); }
  }
  paintResults();
}

// ---------------------------------------------------------------------- the cards

function paintResults() {
  const box = document.getElementById('results');
  if (!box) return;
  const found = narrowed();
  const held = inTab();
  if (!found.length) {
    box.innerHTML = held.length
      ? `<div class="panel empty">Nothing matches. ${
        held.length} games are loaded, so try clearing a filter.</div>`
      : '<div class="panel empty">No games here yet.</div>';
    return;
  }
  const groups = grouped(ordered(found));
  const count = found.length === held.length
    ? `${held.length} games`
    : `${found.length} of ${held.length} games`;
  box.innerHTML = `<div class="muted small u-mb-6">${count}${
    BROWSE.group === 'none' ? '' : ` in ${groups.length} groups`}</div>`
    + pillLegendHtml()
    + groups.map(group => (group.title ? `
      <div class="groupHead">
        <h3>${esc(group.title)}</h3>
        <span class="muted small">${group.games.length} game${
          group.games.length === 1 ? '' : 's'}${
          group.note ? ' &middot; ' + esc(group.note) : ''}</span>
      </div>` : '')
      + `<div class="grid">${group.games.map(cardHtml).join('')}</div>`).join('');

  box.querySelectorAll('.card').forEach(card => {
    card.onclick = () => { pending('Loading the game…'); go('#/game/' + card.dataset.hash); };
  });
}

// What the row of words beside a name is. Every pill already carried its own sentence as
// a hover and nothing said there was anything to point at, so the vocabulary read as the
// maintainer's shorthand -- which, for half of it, is exactly what it is.
function pillLegendHtml() {
  const said = ACCOUNTS
    ? `where it stands${info('level')}, what it scored out of 1${info('index')} and how far
       to trust it${info('trust')}, whether anybody has beaten its strongest computer
       player${info('unbeaten')}, and what the checks found`
    : `what it scored out of 1${info('index')} and how far to trust it${info('trust')}, where
       it is kept${info('committed')}, and what the checks found`;
  return `<div class="muted small u-mb-10">The words under a game's
    name say ${said}. Tap or point at one for what it means.</div>`;
}

function cardHtml(game) {
  // Where a game is kept is the maintainer's question and not a player's. On the site
  // every game is simply here, and "yardstick" is a word from the calibration corpus
  // that says nothing a visitor can act on. The level pill (`levelTag`, Q503b) is the
  // word both copies share.
  const where = ACCOUNTS ? '' : ({ committed: 'yardstick', archived: 'archived',
                                   designed: 'built here', evolved: 'invented here'
                                 }[game.source] || game.source);
  // The pills are always a row of their own. Beside the name they sat next to a short
  // one and dropped under a long one, so the same card was two layouts depending on
  // what the game was called. There is always at least one, because a game with no
  // number still says it has none.
  //
  // One loud thing (V6, S5 of `docs/plan-visual.md`): the opening position says what the
  // game looks like, and the pills are the level and the score with how far to trust it.
  // Every other fact is a quiet line under them, the same words with their hover, no
  // border.
  const notes = unbeatenTag(game) + untriedTag(game) + unplayedTag(game) + fairyTag(game)
    + playableTag(game);
  return `
    <div class="panel card ${styleClass(paperSheet(game.hash))}" data-hash="${esc(game.hash)}">
      ${thumbHtml(game)}
      <div class="cardText">
      <h2>${esc(game.name)}</h2>
      <div class="row pills">${levelTag(game)}${game.preview ? '' : scoreTag(game)
        + trustTag(game.trust, game.status)}</div>
      ${notes ? `<div class="row notes">${notes}</div>` : ''}
      ${exhibitHtml(game.exhibit)}${previewHtml(game.preview)}
      <div class="row muted small u-gap-8">
        <span>${esc(game.board)}</span><span>&middot;</span>
        <span>${game.pieceTypes} piece type${game.pieceTypes === 1 ? '' : 's'}</span>
        ${where ? `<span>&middot;</span><span>${esc(where)}</span>` : ''}
      </div>
      ${peopleHtml(game)}
      <div class="muted small kind" title="${esc((game.cellBins || []).join(', '))}">${
        (game.cellBins || []).slice(0, 4).map(esc).join(' &middot; ')
      }</div>
      <p>${esc(game.description)}</p>
      ${game.preview ? '' : cardSplitHtml(game) + cardSeatHtml(game) + sortedByHtml(game)
        + cardBandHtml(game)}
      ${game.descent ? `<div class="muted small u-mt-6">${
        game.descent} version${game.descent === 1 ? '' : 's'} on from ${
        esc((game.family || {}).name || nameOf((game.family || {}).hash) || 'an earlier game')
      }</div>` : ''}
      </div>
    </div>`;
}

// A card's thumbnail (V6, `docs/plan-visual.md` §3.3): the game's opening position, from
// the listing's `still`, drawn by the still board every other preview uses. The listing
// sends each piece's drawing once (`pieceArt`) and the card names it by its stem, so the
// drawing is put back here. The game's own hash goes in, so its wood is its own (Q588's
// grain is hashed from it); no hand strip and no marks: a card is a glance, not the rules.
//
// The drawings come with the listing, so anything that draws a thumbnail from a listing
// first takes that listing's drawings (`takePieceArt`): the games page, the landing page's
// board and its *Made by members* shelf. Q762: the shelf drew before the landing page had
// taken them, so a first visit's shelf showed every piece as its letter.
function thumbHtml(game) {
  const still = game && game.still;
  if (!still) return '';
  const icons = (still.icons || []).map(pair => (pair || []).map(icon =>
    icon && PIECE_ART[icon.stem]
      ? { href: PIECE_ART[icon.stem], rotate: icon.rotate, stem: icon.stem } : null));
  return `<div class="thumb" aria-hidden="true">${boardPreview(Object.assign({}, still,
    { icons: icons, hash: game.hash, handsUsed: false, plain: true }), null)}</div>`;
}

// The listing's drawings, stem to drawing, for every thumbnail drawn from it. A listing
// without them (none arrived) leaves the drawings already taken.
function takePieceArt(data) {
  if (data && data.pieceArt) PIECE_ART = data.pieceArt;
}

// The store's tag, and the honest blank beside it. Three states, and the third is half
// the library: thirty of the sixty-one games have no picker ladder, so they have no
// computer players to be unbeaten against. Saying nothing on those cards would read as
// "somebody has beaten it", which is the one thing that is certainly not true, so they
// say what they are and the board offers them as practice games (B17; *practice only*
// is `docs/archive/plan-launch.md` §3.5's word for it, L5).
function unbeatenTag(game) {
  if (game.rated === false) {
    return `<span class="tag" title="Practice only: its computer players have no ratings
      yet, so a game here counts for nothing and there is nobody to beat. You can still
      play it, with undo and restart.">unrated</span>`.replace(/\s+/g, ' ');
  }
  if (game.unbeaten === true) {
    // Never an agent string (B15): the sentence names no player, since *its strongest
    // computer player* is what a visitor knows it by.
    return `<span class="tag score" title="Nobody has beaten its strongest computer player
      yet. The first to do it is named on the game for good.">Unbeaten</span>`
      .replace(/\s+/g, ' ');
  }
  if (game.unbeaten === false && game.firstClear) {
    return `<span class="tag" title="First beaten by ${esc(game.firstClear.handle)}${
      game.firstClear.when ? ' on ' + esc(game.firstClear.when.slice(0, 10)) : ''}.">beaten
      by ${game.clears} ${game.clears === 1 ? 'person' : 'people'}</span>`;
  }
  return '';
}

// A mark for a game the signed-in viewer has not finished a rated game on yet.
// `game.untried` is null for a visitor and false for anybody who has, so only an
// explicit `true` from the store draws it (box B2 of `docs/archive/plan-meta-game.md`).
function untriedTag(game) {
  if (game.untried !== true) return '';
  return `<span class="tag" title="You have not finished a rated game here yet. A
    practice game leaves no mark, so this stays until a rated one ends.">not yet
    tried</span>`.replace(/\s+/g, ' ');
}

// A mark for a game with no finished rated game on it at all -- M5's target, one
// human claim waiting to be made. `humanGames` is undefined on the local application,
// where there is no store to ask, so only an explicit zero draws it.
function unplayedTag(game) {
  if (game.humanGames !== 0) return '';
  return `<span class="tag" title="Nobody has finished a rated game on this yet. Be the
    first.">not played here yet</span>`.replace(/\s+/g, ' ');
}

// What people made of it, where the store knows. Stars and human games are two of §5's
// three orderings and both are absent rather than nought on a game nobody has touched:
// "nobody has voted" is not "everybody gave it one star".
//
// A line of its own rather than more items on the line above, which already carries
// three and wraps on a narrow card -- and a separator left dangling at the end of a
// wrapped line is the untidiest way to say two extra things.
function peopleHtml(game) {
  const bits = [];
  if (game.votes) {
    bits.push(`<span title="${game.votes} vote${game.votes === 1 ? '' : 's'}">&#9733; ${
      game.stars.toFixed(1)}</span>`);
  }
  if (game.humanGames) {
    bits.push(`<span>${game.humanGames} game${game.humanGames === 1 ? '' : 's'} played by
      people</span>`);
  }
  if (!bits.length) return '';
  return `<div class="row muted small u-gap-8 u-mt-2">${
    bits.join('<span>&middot;</span>')}</div>`;
}

// When a game's runs did not agree about it. Four runs of one game can reach four
// verdicts, and on today's library four games split -- so a card showing one verdict over
// a number owes the reader the split that produced it. The sentence is the server's and
// the refusing runs' own words follow it, the same as on the game's page.
//
// Drawn under every ordering, because "its runs disagree" is not a fact about how the
// list is currently sorted. A game whose runs agreed, and every one-run score, draws
// nothing at all.
// The split on the face of the card and the refusing runs' own words behind it: *which*
// run refused is the game's page to spell out, and a card is a thing you choose from.
function cardSplitHtml(game) {
  const said = splitSentences(game.runVerdict, game.runs);
  if (!said.length) return '';
  return `<div class="muted small u-mt-6" title="${esc(said.join(' '))}">${
    esc(said[0])}</div>`;
}

// How far this game leans toward one seat, on the card, when the seat check tolerated it.
// Drawn under every ordering for the reason the split is: a game ranked with a five-in-six
// second-seat lean is a thing to know before you pick it, whatever the list is sorted by.
function cardSeatHtml(game) {
  const said = seatLeanText(game);
  if (!said) return '';
  const band = game.seat.bands && game.seat.bands.top_rung_seat;
  const short = band && band.favoured
    ? `leans to the ${band.favoured} side, ${Math.round(band.share * 100)}% of decided games`
    : 'leans toward one side';
  return `<div class="muted small u-mt-6" title="${esc(said)}">${
    esc(short)}</div>`;
}

// Where this game sits against every other measured game, on the card, while the list is
// ordered by the index. It belongs to that ordering the way `sortedByHtml` belongs to a
// component ordering: a list arranged by a number should say what that number buys, and
// what it buys here is three groups rather than a place. Under any other ordering the
// band is beside the point and the card stays short.
//
// `game.band` is absent on a game whose measurement is stale or missing, so those cards
// draw nothing under the index ordering either. `bandHtml` is `core.py`'s and the game's
// own page calls the same one.
function cardBandHtml(game) {
  return BROWSE.sort === 'index' ? bandHtml(game.band) : '';
}

// The measurement the list is currently ordered by, on every card in it. A ranking whose
// criterion is invisible is one you have to take on trust, and this is also the only
// place a single component's number is ever shown outside the score panel.
function sortedByHtml(game) {
  const key = BROWSE.sort;
  // Q826: ordered by the day each game joined, every card says its day.
  if (key === 'added') {
    return `<div class="metric small muted">${game.added
      ? 'added ' + esc(dayWords(game.added)) : 'no day recorded'}</div>`;
  }
  if (!key.includes(':')) return '';
  const [kind, name] = [key.slice(0, key.indexOf(':')), key.slice(key.indexOf(':') + 1)];
  const legend = ((METRICS || {})[kind === 'family' ? 'families' : 'components'] || {})[name];
  const value = sortValue(game, key);
  const title = legend ? legendTitle(legend, name) : name;
  if (value === null) {
    return `<div class="metric">${esc(title)}
      <span class="muted">not measured</span></div>`;
  }
  // A component is a mean over the same runs the score is, so the count belongs here
  // too. It sits in the hover rather than on the line, because the score pill above
  // already says it on the face of the card and twice is clutter.
  const rested = runsPhrase(game.seeds);
  return `<div class="metric" title="${esc(((legend && legendSays(legend)) || '')
      + (rested ? ` Taken as the ${rested}.` : ''))}">
      <span>${esc(title)}</span>
      <span class="bar"><div class="${styleClass(`width:${Math.round(value * 100)}%`)}"></div></span>
      <b>${value.toFixed(2)}</b>
    </div>`;
}
