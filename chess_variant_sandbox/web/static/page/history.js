// How the scores have moved: every score the library has kept for its games, drawn
// (SH3 of `docs/plan-score-history.md`, §2.1). One call to `GET /api/library/history`
// (`history.library_history`) and five views of what it sends, sharing one pick state.
//
// **A picture of the record, never a score.** Every number printed is one a stored file
// holds, as stored. The page works out three things and says so on screen: where a point
// goes, a game's rank within one column, and a column's median. It works out no index, no
// interval and no verdict, and draws no line between two weights versions' numbers that
// says one is better than the other (D5).
//
// **A provisional score is never a point** (`library.Game.status`): the payload puts it in
// a game's `current`, marked, and never in a column; the table and the note name it with
// its revision and its date, as a provisional number is shown everywhere else.
//
// Colours are tokens (§2.4): `--pick-1` to `--pick-3` follow a pick, not a rank, for as
// long as the game is picked (D4); `--trace` is every game not picked; the band of view 2
// is `--pick-1` at an opacity. Fixed declarations are classes in `style.py`; the tooltip's
// place is a CSSOM write, which the page's policy allows.

const HISTORY_PATH = '/score-history';
const HISTORY_PICKS_MAX = 3;
const HISTORY_SVG = 'http://www.w3.org/2000/svg';
// Ticks every 0.2 from 0.2 to 1.0 (§2.4); a number under the floor is drawn at the floor.
const HISTORY_TICKS = [0.2, 0.4, 0.6, 0.8, 1.0];
// Below this width the bump chart drops its end labels (§2.5).
const HISTORY_NARROW = 640;
// The four groups a chip is filed under (§2.1), in their order.
const HISTORY_GROUPS = [
  ['cohort', 'Scored at every snapshot'],
  ['committed', 'Other committed rulesets'],
  ['archived', 'Archived games'],
  ['preview', 'Previews'],
];

let HISTORY_DATA = null;
// The picked games' hashes, oldest pick first, and each one's slot (0, 1, 2): a pick keeps
// its hue while it is picked, and the fourth pick drops the oldest (D4).
let HISTORY_PICKED = [];
let HISTORY_SLOT = {};
let HISTORY_RESIZE = null;

function onHistoryPath() {
  return location.pathname.replace(/\/+$/, '') === HISTORY_PATH;
}

async function showHistory() {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  view.innerHTML = `
    <div class="history" id="history">
      <h2>How the scores have moved</h2>
      <div id="historyLead" class="historyLead"></div>
      <div id="historyBody"></div>
    </div>`;
  pending('Reading the scores the library has kept…');
  const data = await api('/api/library/history');
  clearPending();
  HISTORY_DATA = historyPrepared(data);
  historyStartPicks();
  document.getElementById('tagline').textContent = historySummary(HISTORY_DATA);
  document.getElementById('historyLead').innerHTML = historyLeadHtml(HISTORY_DATA);
  document.getElementById('historyBody').innerHTML = historyBodyHtml(HISTORY_DATA);
  historyWire();
  historyPaint();
  if (!HISTORY_RESIZE) {
    HISTORY_RESIZE = () => {
      clearTimeout(historyDraw.timer);
      historyDraw.timer = setTimeout(() => {
        if (document.getElementById('history')) historyDraw();
      }, 120);
    };
    window.addEventListener('resize', HISTORY_RESIZE);
  }
}

// ----------------------------------------------------------------- the payload, read

// The payload with the lookups the views share: each snapshot's label, the games by hash,
// and which group each game's chip is filed under. Nothing here is a number of its own.
function historyPrepared(data) {
  const snapshots = data.snapshots || [];
  const seen = {};
  const labelled = snapshots.map(snapshot => {
    const version = snapshot.weights_version || snapshot.id;
    // A version with two columns (v12 at its adoption, from the boards, and v12's last
    // score a game, from the ledgers) says which is the later.
    const label = seen[version] ? `${version} late` : String(version);
    seen[version] = true;
    return { ...snapshot, label };
  });
  const cohort = new Set(data.cohort || []);
  const games = (data.games || []).map(game => ({
    ...game,
    group: cohort.has(game.hash) ? 'cohort' : game.source,
  }));
  const byHash = {};
  games.forEach(game => { byHash[game.hash] = game; });
  return {
    current: data.currentVersion,
    snapshots: labelled,
    ids: labelled.map(snapshot => snapshot.id),
    games,
    byHash,
    cohort: (data.cohort || []).filter(hash => byHash[hash]),
    joins: data.joins || [],
  };
}

// The library's own games: the ones a band, a median and a trace are about. A preview is
// drawn only when it is picked.
function historyLibrary(data) {
  return data.games.filter(game => game.source !== 'preview');
}

function historyPoint(game, id) {
  const point = game.points && game.points[id];
  return point && typeof point.index === 'number' ? point : null;
}

function historyFigure(value) {
  return typeof value === 'number' ? value.toFixed(3) : '–';
}

const HISTORY_MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
  'August', 'September', 'October', 'November', 'December'];

function historyDay(text, long) {
  const found = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(text || ''));
  if (!found) return '';
  const month = HISTORY_MONTHS[Number(found[2]) - 1] || '';
  const day = Number(found[3]);
  return long ? `${day} ${month} ${found[1]}` : `${day} ${month.slice(0, 3)}`;
}

// Days since the epoch, for the day-by-day axis: a date is placed, not computed with.
function historyDayNumber(text) {
  const found = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(text || ''));
  return found ? Math.round(Date.UTC(+found[1], +found[2] - 1, +found[3]) / 86400000) : null;
}

function historyMedian(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function historySummary(data) {
  const first = data.snapshots[0];
  return `${data.games.length} games over ${data.snapshots.length} snapshots${
    first ? `, from ${first.label} on ${historyDay(first.date, true)}` : ''}`;
}

// ----------------------------------------------------------------- picks

// The picks the address names (`?pick=<hash>,...`), else three to start with: the
// long-lived games highest, in the middle and lowest in the current column, so the first
// screen shows the spread. A pick never reorders another's hue.
function historyStartPicks() {
  HISTORY_PICKED = [];
  HISTORY_SLOT = {};
  const asked = new URLSearchParams(location.search).get('pick');
  const named = String(asked || '').split(',').filter(hash => HISTORY_DATA.byHash[hash]);
  const start = named.length ? named : historyOpeningPicks(HISTORY_DATA);
  start.slice(0, HISTORY_PICKS_MAX).forEach(hash => historyTake(hash));
}

function historyOpeningPicks(data) {
  // The newest column with a point in it: while every current score is provisional (a new
  // core, a new analysis) the current column is empty, and so is the cohort, so the
  // library's games with a point in that column stand in for it.
  const id = [...data.ids].reverse().find(one =>
    data.games.some(game => historyPoint(game, one))) || data.ids[data.ids.length - 1];
  const pool = data.cohort.length ? data.cohort
    : historyLibrary(data).map(game => game.hash);
  const ranked = historyRanked(pool, data, id);
  if (ranked.length <= HISTORY_PICKS_MAX) return ranked;
  return [ranked[0], ranked[Math.floor((ranked.length - 1) / 2)], ranked[ranked.length - 1]];
}

function historyTake(hash) {
  if (HISTORY_SLOT[hash] !== undefined) return;
  if (HISTORY_PICKED.length >= HISTORY_PICKS_MAX) {
    const oldest = HISTORY_PICKED.shift();
    delete HISTORY_SLOT[oldest];
  }
  const used = new Set(Object.values(HISTORY_SLOT));
  let slot = 0;
  while (used.has(slot)) slot += 1;
  HISTORY_PICKED.push(hash);
  HISTORY_SLOT[hash] = slot;
}

function historyToggle(hash) {
  if (HISTORY_SLOT[hash] !== undefined) {
    HISTORY_PICKED = HISTORY_PICKED.filter(one => one !== hash);
    delete HISTORY_SLOT[hash];
  } else {
    historyTake(hash);
  }
  // The picks travel with the address, so a link shows what its sender saw.
  try {
    const query = HISTORY_PICKED.length ? '?pick=' + HISTORY_PICKED.join(',') : '';
    history.replaceState(null, '', location.pathname + query + location.hash);
    RENDERED = location.href;
  } catch (error) { /* the address stays */ }
  historyPaint();
}

function historyHue(hash) {
  const slot = HISTORY_SLOT[hash];
  return slot === undefined ? 'var(--trace)' : `var(--pick-${slot + 1})`;
}

function historySlotClass(hash) {
  const slot = HISTORY_SLOT[hash];
  return slot === undefined ? '' : ` historyPick${slot + 1}`;
}

// ----------------------------------------------------------------- the page's words

function historyLeadHtml(data) {
  const first = data.snapshots[0];
  const last = data.snapshots[data.snapshots.length - 1];
  const span = first && last
    ? `across ${data.snapshots.length} snapshots${info('snapshot')} of the scorer, from
       ${esc(first.label)} of its rules on ${esc(historyDay(first.date, true))} to ${
       esc(last.label)}`
    : 'across every snapshot of the scorer';
  return `<p>Every score the library has kept for its games, ${span}. Each version of
      the scoring rules${info('weightsVersion')} sets its own scale, so a level is only comparable
      within a column. The shape of a game's line and its rank among the others carry across
      versions; the absolute number does not.</p>
    <p class="muted small">Every number here is one a stored file holds, printed as stored.
      The page works out only where each point goes, each game's rank within one column, and
      a column's median. A score measured under an earlier revision of the scorer is
      provisional${info('historyProvisional')} and is never drawn as a point.</p>`;
}

function historyBodyHtml(data) {
  const groups = HISTORY_GROUPS.map(([key, words]) => {
    const games = historyGroup(data, key);
    if (!games.length) return '';
    const chips = `<div class="chips" data-history-group="${key}"></div>`;
    // The long-lived games are open; the other groups fold, so the views stay near the top.
    return key === 'cohort'
      ? `<div class="historyGroup"><h4>${esc(words)} <span class="muted">${
          games.length}</span></h4>${chips}</div>`
      : `<details class="historyGroup"><summary>${esc(words)} <span class="muted">${
          games.length}</span></summary>${chips}</details>`;
  }).join('');
  const card = (key, title, said) => `
    <section class="panel historyCard" id="history-${key}">
      <div class="historyHead"><h3>${title}</h3><span class="historyLegend"
        data-history-legend="${key}"></span></div>
      <p class="muted small">${said}</p>
      <div class="historyPlot" data-history-plot="${key}"></div>
    </section>`;
  return `
    <section class="panel historyCard">
      <div class="historyHead"><h3>Pick up to three games</h3><span class="historyLegend"
        data-history-legend="picks"></span></div>
      <p class="muted small">A picked game is drawn in colour in every view below, and a
        fourth pick drops the oldest. Picked to start: the long-lived games highest, in the
        middle and lowest in the newest column with scores in it.</p>
      ${groups}
    </section>
    ${card('index', 'The score under each version of the scoring rules',
      `Every library game a faint line, the picked ones in colour with their last value at
       the line's end. A hollow marker is a score that was refused${info('refusedScore')}. A
       dotted stretch crosses an edit of the game's rules. Point at a column for its values
       and the library's median.`)}
    ${card('band', 'The library band under each version',
      `The lowest, the median and the highest score of the library's games that were not
       refused, at each snapshot, with how many there were. It shows what each version did to
       the scale, not what it did to any one game.`)}
    ${card('rank', `Rank among the ${data.cohort.length} long-lived games`,
      `The games with a score at every snapshot, ranked within each column, 1 the highest.
       Rank does not depend on a version's scale, so it is the one reading this page carries
       across versions.`)}
    ${card('days', 'Day by day under one version',
      `The score each game carried while one version of the scoring rules was current, stepping where a
       new measurement replaced it: a replay, a new analysis, new players. Under the current
       version the ring at the right is the score each game has now.`)}
    <section class="panel historyCard" id="history-table">
      <div class="historyHead"><h3>Every number</h3></div>
      <div class="historyTable" tabindex="0"><table id="historyTable"></table></div>
      <div class="muted small historyNote" id="historyNote"></div>
    </section>`;
}

function historyGroup(data, key) {
  const games = key === 'cohort'
    ? data.cohort.map(hash => data.byHash[hash])
    : data.games.filter(game => game.group === key);
  return games.slice().sort((a, b) => a.name.localeCompare(b.name));
}

function historyWire() {
  const body = document.getElementById('historyBody');
  body.addEventListener('click', event => {
    const chip = event.target.closest && event.target.closest('[data-history-pick]');
    if (chip) historyToggle(chip.dataset.historyPick);
  });
}

// Everything a pick changes: the chips, the legends, the views and the table.
function historyPaint() {
  const data = HISTORY_DATA;
  HISTORY_GROUPS.forEach(([key]) => {
    const box = document.querySelector(`[data-history-group="${key}"]`);
    if (!box) return;
    box.innerHTML = historyGroup(data, key).map(game => {
      const picked = HISTORY_SLOT[game.hash] !== undefined;
      return `<button type="button" class="chip historyChip${historySlotClass(game.hash)}"
        data-history-pick="${esc(game.hash)}" aria-pressed="${picked}">${esc(game.name)}</button>`;
    }).join('');
  });
  const picks = HISTORY_PICKED.map(hash => `<span class="historyKey${
    historySlotClass(hash)}">${esc(data.byHash[hash].name)}</span>`).join('');
  document.querySelectorAll('[data-history-legend]').forEach(legend => {
    const key = legend.dataset.historyLegend;
    const extra = {
      index: '<span class="historyKey historyTrace">other library games</span>'
        + '<span class="historyKey historyHollow">refused</span>',
      band: '<span class="historyKey historyPick1">median</span>'
        + '<span class="historyKey historyBand">lowest to highest</span>',
      rank: '<span class="historyKey historyTrace">the other long-lived games</span>',
      days: '<span class="historyKey historyTrace">other library games</span>'
        + '<span class="historyKey historyHollow">the score now</span>',
    }[key] || '';
    legend.innerHTML = (key === 'band' ? '' : picks) + extra;
  });
  historyDraw();
}

function historyDraw() {
  const data = HISTORY_DATA;
  if (!data) return;
  historyIndexView(data);
  historyBandView(data);
  historyRankView(data);
  historyDaysView(data);
  historyTableView(data);
}

// ----------------------------------------------------------------- drawing helpers

function historyEl(tag, attrs, text) {
  const node = document.createElementNS(HISTORY_SVG, tag);
  Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function historyFrame(key, height) {
  const plot = document.querySelector(`[data-history-plot="${key}"]`);
  if (!plot) return null;
  plot.textContent = '';
  const width = Math.max(300, Math.round(plot.clientWidth || 300));
  const svg = historyEl('svg', {
    viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', class: 'historySvg',
  });
  plot.append(svg);
  const tip = document.createElement('div');
  tip.className = 'historyTip';
  tip.hidden = true;
  plot.append(tip);
  return { plot, svg, tip, width, height };
}

function historyText(svg, x, y, words, attrs) {
  svg.append(historyEl('text', {
    x: x.toFixed(1), y: y.toFixed(1), 'font-size': 11.5, fill: 'var(--muted)',
    class: 'historyFigures', ...(attrs || {}),
  }, words));
}

function historyGrid(frame, y, margin) {
  HISTORY_TICKS.forEach(tick => {
    frame.svg.append(historyEl('line', {
      x1: margin.l, x2: frame.width - margin.r, y1: y(tick), y2: y(tick),
      stroke: 'var(--line)', 'stroke-width': 1,
    }));
    historyText(frame.svg, margin.l - 7, y(tick) + 4, tick.toFixed(1), { 'text-anchor': 'end' });
  });
}

function historyScale(height, margin) {
  const low = HISTORY_TICKS[0], high = HISTORY_TICKS[HISTORY_TICKS.length - 1];
  return value => {
    const clipped = Math.max(low, Math.min(high, value));
    return margin.t + (high - clipped) / (high - low) * (height - margin.t - margin.b);
  };
}

// 8 px markers with a 2 px ring of the panel's colour; a refused one is hollow (§2.4).
function historyMarker(svg, x, y, hue, hollow) {
  svg.append(historyEl('circle', {
    cx: x.toFixed(1), cy: y.toFixed(1), r: 4,
    fill: hollow ? 'var(--panel)' : hue, stroke: hollow ? hue : 'var(--panel)', 'stroke-width': 2,
  }));
}

function historyPath(points) {
  return points.map((p, at) => `${at ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
}

// End labels for the picks, pushed apart so that two never overlap.
function historyEndLabels(svg, labels) {
  labels.sort((a, b) => a.y - b.y);
  for (let at = 1; at < labels.length; at += 1) {
    if (labels[at].y - labels[at - 1].y < 14) labels[at].y = labels[at - 1].y + 14;
  }
  labels.forEach(label => historyText(svg, label.x, label.y + 4, label.words,
    { fill: 'var(--ink)', 'font-weight': 600 }));
}

function historyTipAt(frame, html, x, event) {
  const box = frame.svg.getBoundingClientRect();
  frame.tip.innerHTML = html;
  frame.tip.hidden = false;
  const scale = box.width / frame.width;
  const wide = frame.tip.offsetWidth;
  let left = x * scale + 14;
  if (left + wide > box.width) left = x * scale - wide - 14;
  frame.tip.style.left = Math.max(0, left) + 'px';
  frame.tip.style.top = Math.max(0, event.clientY - box.top - 10) + 'px';
}

function historyTipRow(hash, words, figure) {
  return `<div class="historyTipRow"><span class="historyKey${historySlotClass(hash)}">${
    esc(words)}</span><span class="historyFigures">${esc(figure)}</span></div>`;
}

// The column a pointer is over, by its place across the plot.
function historyColumnAt(frame, event, x0, step, count) {
  const box = frame.svg.getBoundingClientRect();
  const sx = (event.clientX - box.left) * frame.width / box.width;
  return Math.max(0, Math.min(count - 1, Math.round((sx - x0) / step)));
}

function historyHover(frame, column, onMove) {
  const cross = column ? historyEl('line', {
    y1: column.top, y2: column.bottom, stroke: 'var(--muted)', 'stroke-width': 1,
    visibility: 'hidden',
  }) : null;
  if (cross) frame.svg.append(cross);
  frame.svg.addEventListener('mousemove', event => {
    const x = onMove(event);
    if (cross && x !== null) {
      cross.setAttribute('x1', x);
      cross.setAttribute('x2', x);
      cross.setAttribute('visibility', 'visible');
    }
  });
  frame.svg.addEventListener('mouseleave', () => {
    frame.tip.hidden = true;
    if (cross) cross.setAttribute('visibility', 'hidden');
  });
}

// The library's games that were not refused in one column, and their values.
function historyViable(data, id) {
  return historyLibrary(data)
    .map(game => [game, historyPoint(game, id)])
    .filter(([, point]) => point && point.viable !== false);
}

// ----------------------------------------------------------------- 1. by version

function historyIndexView(data) {
  const height = 400;
  const frame = historyFrame('index', height);
  if (!frame) return;
  const narrow = frame.width < HISTORY_NARROW;
  const margin = { l: 38, r: narrow ? 64 : 150, t: 12, b: 44 };
  const ids = data.ids;
  const step = (frame.width - margin.l - margin.r) / Math.max(1, ids.length - 1);
  const x = at => margin.l + at * step;
  const y = historyScale(height, margin);
  historyGrid(frame, y, margin);
  historyColumnLabels(frame.svg, data, x, height - margin.b);
  const drawn = historyLibrary(data).filter(game => HISTORY_SLOT[game.hash] === undefined);
  drawn.forEach(game => historyLine(frame.svg, game, ids, x, y, false));
  const labels = [];
  HISTORY_PICKED.forEach(hash => {
    const game = data.byHash[hash];
    const end = historyLine(frame.svg, game, ids, x, y, true);
    if (end) {
      labels.push({ x: end[0] + 9, y: end[1],
        words: narrow ? historyFigure(end[2]) : `${game.name} ${historyFigure(end[2])}` });
    }
  });
  historyEndLabels(frame.svg, labels);
  historyHover(frame, { top: margin.t, bottom: height - margin.b }, event => {
    const at = historyColumnAt(frame, event, margin.l, step, ids.length);
    const snapshot = data.snapshots[at];
    const values = historyViable(data, snapshot.id).map(([, point]) => point.index);
    const rows = HISTORY_PICKED.map(hash => {
      const point = historyPoint(data.byHash[hash], snapshot.id);
      return historyTipRow(hash, data.byHash[hash].name, point
        ? historyFigure(point.index) + (point.viable === false ? ' refused' : '') : '–');
    }).join('');
    historyTipAt(frame, `<b>${esc(snapshot.label)}</b> <span class="muted">${
      esc(historyDay(snapshot.date, true))}</span><div class="muted">${esc(snapshot.basis || '')}</div>${
      rows}<div class="historyTipRow muted"><span>library median</span><span class="historyFigures">${
      historyFigure(historyMedian(values))} of ${values.length}</span></div>`, x(at), event);
    return x(at);
  });
}

function historyColumnLabels(svg, data, x, base) {
  const narrow = data.snapshots.length > 1 && x(1) - x(0) < 46;
  data.snapshots.forEach((snapshot, at) => {
    // On a phone every other column's date goes, so the labels do not run together.
    historyText(svg, x(at), base + 17, snapshot.label.replace(' late', '+'),
      { 'text-anchor': 'middle', fill: 'var(--ink)', 'font-weight': 600,
        'font-size': narrow ? 10 : 11.5 });
    if (!narrow || at % 2 === 0) {
      historyText(svg, x(at), base + 31, historyDay(snapshot.date),
        { 'text-anchor': 'middle', 'font-size': 10.5 });
    }
  });
}

// One game's line across the columns: a gap where it has no point, dotted where it crosses
// an edit of its rules (D2), markers and a hollow refused one for a pick. Its last point.
function historyLine(svg, game, ids, x, y, picked) {
  const hue = historyHue(game.hash);
  const points = ids.map((id, at) => {
    const point = historyPoint(game, id);
    return point ? { at, xy: [x(at), y(point.index)], point } : null;
  });
  for (let at = 1; at < points.length; at += 1) {
    const one = points[at - 1], two = points[at];
    if (!one || !two) continue;
    const crosses = one.point.how === 'file'
      && (two.point.how !== 'file' || two.point.as !== one.point.as);
    svg.append(historyEl('path', {
      d: historyPath([one.xy, two.xy]), fill: 'none', stroke: hue,
      'stroke-width': picked ? 2 : 1.25, 'stroke-linecap': 'round',
      ...(crosses ? { 'stroke-dasharray': '2 4' } : {}),
    }));
  }
  if (!picked) return null;
  points.forEach(one => {
    if (one) historyMarker(svg, one.xy[0], one.xy[1], hue, one.point.viable === false);
  });
  const last = points.filter(Boolean).pop();
  return last ? [last.xy[0], last.xy[1], last.point.index] : null;
}

// ----------------------------------------------------------------- 2. the band

function historyBandView(data) {
  const height = 300;
  const frame = historyFrame('band', height);
  if (!frame) return;
  const margin = { l: 38, r: 16, t: 12, b: 44 };
  const ids = data.ids;
  const step = (frame.width - margin.l - margin.r) / ids.length;
  const x = at => margin.l + (at + 0.5) * step;
  const y = historyScale(height, margin);
  historyGrid(frame, y, margin);
  const columns = ids.map(id => {
    const rows = historyViable(data, id).sort((a, b) => a[1].index - b[1].index);
    return {
      count: rows.length,
      floor: rows[0], top: rows[rows.length - 1],
      median: historyMedian(rows.map(([, point]) => point.index)),
    };
  });
  const narrow = ids.length * 46 > frame.width - margin.l - margin.r;
  columns.forEach((column, at) => {
    historyText(frame.svg, x(at), height - margin.b + 17,
      data.snapshots[at].label.replace(' late', '+'),
      { 'text-anchor': 'middle', fill: 'var(--ink)', 'font-weight': 600 });
    if (!column.count) return;
    frame.svg.append(historyEl('line', {
      x1: x(at), x2: x(at), y1: y(column.top[1].index), y2: y(column.floor[1].index),
      stroke: 'var(--pick-1)', 'stroke-width': 8, 'stroke-linecap': 'round', opacity: 0.3,
    }));
    historyMarker(frame.svg, x(at), y(column.median), 'var(--pick-1)', false);
    if (!narrow || at % 2 === 0) {
      historyText(frame.svg, x(at), height - margin.b + 31, `${column.count}`,
        { 'text-anchor': 'middle', 'font-size': 10.5 });
    }
  });
  historyHover(frame, null, event => {
    const at = historyColumnAt(frame, event, x(0), step, ids.length);
    const column = columns[at];
    if (!column.count) return null;
    const row = (words, value) => `<div class="historyTipRow"><span>${words}</span><span
      class="historyFigures">${value}</span></div>`;
    historyTipAt(frame, `<b>${esc(data.snapshots[at].label)}</b> <span class="muted">${
      esc(historyDay(data.snapshots[at].date, true))}</span>${
      row('highest', `${esc(column.top[0].name)} ${historyFigure(column.top[1].index)}`)}${
      row('median', historyFigure(column.median))}${
      row('lowest', `${esc(column.floor[0].name)} ${historyFigure(column.floor[1].index)}`)}${
      row('<span class="muted">games not refused</span>', column.count)}`, x(at), event);
    return null;
  });
}

// ----------------------------------------------------------------- 3. rank

// The games ranked within one column, highest first; a tie keeps the names' order.
function historyRanked(hashes, data, id) {
  return hashes
    .filter(hash => historyPoint(data.byHash[hash], id))
    .slice()
    .sort((a, b) => historyPoint(data.byHash[b], id).index - historyPoint(data.byHash[a], id).index
      || data.byHash[a].name.localeCompare(data.byHash[b].name));
}

function historyRankView(data) {
  const count = data.cohort.length;
  const plot = document.querySelector('[data-history-plot="rank"]');
  if (!plot) return;
  if (!count) {
    const said = document.createElement('p');
    said.className = 'muted small';
    said.textContent = 'No game has a score at every snapshot right now: the newest column'
      + ' is empty while the current scores are provisional.';
    plot.textContent = '';
    plot.append(said);
    return;
  }
  const narrow = (plot.clientWidth || 300) < HISTORY_NARROW;
  const row = 21;
  const margin = { l: narrow ? 14 : 150, r: narrow ? 14 : 150, t: 12, b: 44 };
  const height = margin.t + margin.b + row * (count - 1) + 10;
  const frame = historyFrame('rank', height);
  const ids = data.ids;
  const step = (frame.width - margin.l - margin.r) / Math.max(1, ids.length - 1);
  const x = at => margin.l + at * step;
  const y = rank => margin.t + 5 + (rank - 1) * row;
  const ranks = {};
  ids.forEach(id => historyRanked(data.cohort, data, id).forEach((hash, at) => {
    (ranks[hash] = ranks[hash] || {})[id] = at + 1;
  }));
  for (let rank = 1; rank <= count; rank += 1) {
    if (rank === 1 || rank % 5 === 0) {
      frame.svg.append(historyEl('line', {
        x1: margin.l, x2: frame.width - margin.r, y1: y(rank), y2: y(rank),
        stroke: 'var(--line)', 'stroke-width': 1,
      }));
    }
  }
  historyColumnLabels(frame.svg, data, x, height - margin.b);
  const draw = (hash, picked) => {
    const hue = historyHue(hash);
    const points = ids.map((id, at) => [x(at), y(ranks[hash][id])]);
    frame.svg.append(historyEl('path', {
      d: historyPath(points), fill: 'none', stroke: hue, 'stroke-width': picked ? 2 : 1.25,
      'stroke-linejoin': 'round',
    }));
    if (picked) points.forEach(p => historyMarker(frame.svg, p[0], p[1], hue, false));
    if (narrow) return;
    const style = { fill: picked ? 'var(--ink)' : 'var(--muted)', 'font-weight': picked ? 600 : 400 };
    const name = data.byHash[hash].name;
    historyText(frame.svg, x(0) - 10, points[0][1] + 4, `${ranks[hash][ids[0]]}. ${name}`,
      { 'text-anchor': 'end', ...style });
    historyText(frame.svg, x(ids.length - 1) + 10, points[points.length - 1][1] + 4,
      `${ranks[hash][ids[ids.length - 1]]}. ${name}`, style);
  };
  data.cohort.filter(hash => HISTORY_SLOT[hash] === undefined).forEach(hash => draw(hash, false));
  HISTORY_PICKED.filter(hash => ranks[hash]).forEach(hash => draw(hash, true));
  historyHover(frame, null, event => {
    const box = frame.svg.getBoundingClientRect();
    const at = historyColumnAt(frame, event, margin.l, step, ids.length);
    const sy = (event.clientY - box.top) * frame.height / box.height;
    const rank = Math.max(1, Math.min(count, Math.round((sy - margin.t - 5) / row) + 1));
    const id = ids[at];
    const hash = data.cohort.find(one => ranks[one][id] === rank);
    if (!hash) return null;
    historyTipAt(frame, `<b>${esc(data.byHash[hash].name)}</b> <span class="muted">${
      esc(data.snapshots[at].label)}</span>${historyTipRow('', 'rank', `${rank} of ${count}`)}${
      historyTipRow('', 'score', historyFigure(historyPoint(data.byHash[hash], id).index))}`,
      x(at), event);
    return null;
  });
}

// ----------------------------------------------------------------- 4. day by day

// The versions the ledgers hold, oldest first, and the current one: the choices of view 4.
function historyDayVersions(data) {
  const found = new Set();
  data.games.forEach(game => (game.days || []).forEach(day => {
    if (day.weights_version) found.add(day.weights_version);
  }));
  found.add(data.current);
  const number = word => Number((/v(\d+)/.exec(word) || [0, 0])[1]);
  return [...found].sort((a, b) => number(a) - number(b) || a.localeCompare(b));
}

// One game's steps under one version: each ledger line from the day it was measured, then
// (under the current version) its current score as the end point.
function historySteps(game, version, current) {
  const steps = (game.days || [])
    .filter(day => day.weights_version === version && typeof day.index === 'number')
    .map(day => ({ day: historyDayNumber(day.measured), to: historyDayNumber(day.replaced),
      index: day.index, viable: day.viable, why: day.why }))
    .filter(step => step.day !== null);
  const now = game.current;
  const end = version === current && now && now.status === 'current' && now.revision === current
    && typeof now.index === 'number'
    ? { day: historyDayNumber(now.measured), index: now.index, viable: now.viable } : null;
  return { steps, end };
}

function historyDaysView(data) {
  const plot = document.querySelector('[data-history-plot="days"]');
  if (!plot) return;
  const versions = historyDayVersions(data);
  let select = document.getElementById('historyVersion');
  if (!select) {
    // The current version once its record spans a week; until then the newest version
    // whose record does, since a version adopted days ago has hardly any day by day yet.
    const spans = version => {
      const days = [];
      data.games.forEach(game => {
        const one = historySteps(game, version, data.current);
        one.steps.forEach(step => days.push(step.day));
        if (one.end && one.end.day !== null) days.push(one.end.day);
      });
      return days.length && Math.max(...days) - Math.min(...days) >= 7;
    };
    const opening = spans(data.current) ? data.current
      : versions.filter(spans).pop() || data.current;
    const label = document.createElement('label');
    label.className = 'historyPickVersion muted small';
    label.textContent = 'version of the scoring rules ';
    select = document.createElement('select');
    select.id = 'historyVersion';
    versions.forEach(version => {
      const option = document.createElement('option');
      option.value = version;
      option.textContent = version === data.current ? `${version}, current` : version;
      option.selected = version === opening;
      select.append(option);
    });
    select.addEventListener('change', () => historyDaysView(HISTORY_DATA));
    label.append(select);
    plot.before(label);
  }
  const version = select.value;
  const height = 400;
  const frame = historyFrame('days', height);
  const narrow = frame.width < HISTORY_NARROW;
  const margin = { l: 38, r: narrow ? 64 : 150, t: 12, b: 40 };
  const series = data.games
    .filter(game => game.source !== 'preview' || HISTORY_SLOT[game.hash] !== undefined)
    .map(game => ({ game, ...historySteps(game, version, data.current) }))
    .filter(one => one.steps.length || one.end);
  const days = [];
  series.forEach(one => {
    one.steps.forEach(step => { days.push(step.day); if (step.to !== null) days.push(step.to); });
    if (one.end && one.end.day !== null) days.push(one.end.day);
  });
  if (!days.length) {
    const said = document.createElement('p');
    said.className = 'muted small';
    said.textContent = `No score has been replaced under ${version} yet.`;
    frame.plot.textContent = '';
    frame.plot.append(said);
    return;
  }
  const first = Math.min(...days), last = Math.max(first + 1, ...days);
  const x = day => margin.l + (day - first) / (last - first) * (frame.width - margin.l - margin.r);
  const y = historyScale(height, margin);
  historyGrid(frame, y, margin);
  const spacing = narrow ? 14 : 7;
  for (let day = first; day <= last; day += 1) {
    const date = new Date(day * 86400000).toISOString().slice(0, 10);
    if ((day - first) % spacing !== 0) continue;
    frame.svg.append(historyEl('line', {
      x1: x(day), x2: x(day), y1: height - margin.b, y2: height - margin.b + 5,
      stroke: 'var(--line)', 'stroke-width': 1,
    }));
    historyText(frame.svg, x(day), height - margin.b + 18, historyDay(date),
      { 'text-anchor': 'middle' });
  }
  const valueAt = (one, day) => {
    let value = null;
    one.steps.forEach(step => { if (step.day <= day) value = step.index; });
    if (one.end && one.end.day !== null && one.end.day <= day) value = one.end.index;
    return value;
  };
  const draw = (one, picked) => {
    const hue = historyHue(one.game.hash);
    const steps = one.steps;
    let d = '';
    steps.forEach((step, at) => {
      d += at ? ` H${x(step.day).toFixed(1)} V${y(step.index).toFixed(1)}`
        : `M${x(step.day).toFixed(1)} ${y(step.index).toFixed(1)}`;
    });
    const tail = steps[steps.length - 1];
    const stop = one.end && one.end.day !== null ? one.end.day
      : tail && tail.to !== null ? tail.to : last;
    if (steps.length) d += ` H${x(stop).toFixed(1)}`;
    if (d) {
      frame.svg.append(historyEl('path', {
        d, fill: 'none', stroke: hue, 'stroke-width': picked ? 2 : 1.25,
        'stroke-linejoin': 'round',
      }));
    }
    if (one.end) {
      if (tail) {
        frame.svg.append(historyEl('line', {
          x1: x(stop), x2: x(stop), y1: y(tail.index), y2: y(one.end.index),
          stroke: hue, 'stroke-width': picked ? 2 : 1.25,
        }));
      }
      frame.svg.append(historyEl('line', {
        x1: x(stop), x2: x(last), y1: y(one.end.index), y2: y(one.end.index),
        stroke: hue, 'stroke-width': picked ? 2 : 1.25,
      }));
      frame.svg.append(historyEl('circle', {
        cx: x(last).toFixed(1), cy: y(one.end.index).toFixed(1), r: 4, fill: 'var(--panel)',
        stroke: hue, 'stroke-width': 2,
      }));
    }
    if (!picked) return null;
    steps.forEach((step, at) => {
      if (!at || step.index !== steps[at - 1].index) {
        historyMarker(frame.svg, x(step.day), y(step.index), hue, step.viable === false);
      }
    });
    const final = one.end ? [x(last), one.end.index] : tail ? [x(stop), tail.index] : null;
    return final;
  };
  series.filter(one => HISTORY_SLOT[one.game.hash] === undefined).forEach(one => draw(one, false));
  const labels = [];
  HISTORY_PICKED.forEach(hash => {
    const one = series.find(candidate => candidate.game.hash === hash);
    if (!one) return;
    const end = draw(one, true);
    if (end) {
      labels.push({ x: end[0] + 9, y: y(end[1]),
        words: narrow ? historyFigure(end[1]) : `${one.game.name} ${historyFigure(end[1])}` });
    }
  });
  historyEndLabels(frame.svg, labels);
  historyHover(frame, { top: margin.t, bottom: height - margin.b }, event => {
    const box = frame.svg.getBoundingClientRect();
    const sx = (event.clientX - box.left) * frame.width / box.width;
    const day = Math.max(first, Math.min(last, Math.round(
      first + (sx - margin.l) / (frame.width - margin.l - margin.r) * (last - first))));
    const date = new Date(day * 86400000).toISOString().slice(0, 10);
    const rows = HISTORY_PICKED.map(hash => {
      const one = series.find(candidate => candidate.game.hash === hash);
      return historyTipRow(hash, HISTORY_DATA.byHash[hash].name,
        historyFigure(one ? valueAt(one, day) : null));
    }).join('');
    historyTipAt(frame, `<b>${esc(historyDay(date, true))}</b> <span class="muted">under ${
      esc(version)}</span>${rows}`, x(day), event);
    return x(day);
  });
}

// ----------------------------------------------------------------- 5. every number

function historyTableView(data) {
  const table = document.getElementById('historyTable');
  if (!table) return;
  const rest = data.games.filter(game => !data.cohort.includes(game.hash))
    .sort((a, b) => a.name.localeCompare(b.name));
  const rows = data.cohort.map(hash => data.byHash[hash])
    .sort((a, b) => a.name.localeCompare(b.name)).concat(rest);
  const head = '<thead><tr><th>game</th>' + data.snapshots.map(snapshot => `<th>${
    esc(snapshot.label)}<small>${esc(historyDay(snapshot.date))}</small></th>`).join('')
    + '</tr></thead>';
  const body = rows.map(game => {
    const cells = data.snapshots.map(snapshot => {
      const point = historyPoint(game, snapshot.id);
      if (point) {
        const refused = point.viable === false;
        const joined = point.as ? `, as ${point.as}` : '';
        return `<td class="historyFigures${refused ? ' historyRefused' : ''}" title="${
          esc((refused ? 'refused' : 'not refused') + joined)}">${historyFigure(point.index)}</td>`;
      }
      const now = game.current;
      if (snapshot.kind === 'current' && now && now.status === 'provisional') {
        return `<td class="muted" title="${esc(`provisional: ${historyFigure(now.index)} under ${
          now.revision}, measured ${historyDay(now.measured, true)}`)}">provisional</td>`;
      }
      return '<td class="muted">–</td>';
    }).join('');
    return `<tr class="${historySlotClass(game.hash).trim()}"><td>${esc(game.name)}</td>${cells}</tr>`;
  }).join('');
  table.innerHTML = head + '<tbody>' + body + '</tbody>';
  document.getElementById('historyNote').innerHTML = historyNoteHtml(data);
}

function historyNoteHtml(data) {
  // The boards file's joins (`same_game`), each to a game on this page: one line a game.
  const crossing = new Map(), renamed = new Map();
  data.joins.forEach(join => {
    const game = data.byHash[join.to];
    if (!game) return;
    const was = join.old_name && join.old_name !== game.name ? ` (was ${join.old_name})` : '';
    if (join.how === 'file') crossing.set(game.name, game.name + was);
    else if (was) renamed.set(game.name, game.name + was);
  });
  const sorted = found => [...found.keys()].sort((a, b) => a.localeCompare(b))
    .map(name => found.get(name));
  crossing.forEach((words, name) => renamed.delete(name));
  const crossed = sorted(crossing), kept = sorted(renamed);
  const provisional = data.games.filter(game => game.current && game.current.status === 'provisional')
    .map(game => `${game.name} ${historyFigure(game.current.index)} under ${
      game.current.revision}, measured ${historyDay(game.current.measured, true)}`);
  const list = items => items.map(esc).join('; ');
  return `<p>An underlined value was refused${info('refusedScore')}: the game failed a basic
      test of a working game when it was measured. A dash means no score was stored for it
      then: the game did not exist yet, was not in that campaign, or had no score. A column
      marked late (+ on the charts) is each game's last score under that version, from its
      record of replaced scores.</p>
    ${kept.length ? `<p>Renamed without a change to the rules, and joined by the same
      identity: ${list(kept)}.</p>` : ''}
    ${crossed.length ? `<p>Joined across an edit of the rules, drawn dotted where the line
      crosses it, so the early points are an earlier version of the game:
      ${list(crossed)}.</p>` : ''}
    ${provisional.length ? `<p>Provisional${info('historyProvisional')}, measured under an
      earlier revision of the scorer and drawn nowhere: ${list(provisional)}.</p>` : ''}`;
}

// ----------------------------------------------------------------- one game's own chart
//
// SH4 (`docs/plan-score-history.md` §2.1): the step chart in a game page's *earlier
// scores* fold (`detail.js` `earlierHtml`). The x axis is time; each point is one stored
// measurement of this game on the day it was measured, holding until the next; the version
// of the scoring rules is written beside each change of version, where the line is broken
// rather than joined, since two versions' numbers are on two scales (D5); a refused
// measurement is hollow; the present score is the end point, and only a current one: a
// provisional score is never a point here either.
//
// What it draws from is the caller's choice and the whole of it: the maintainer and an
// admin pass the ledger (`game.history`), a visitor `for_visitors`' lines
// (`game.earlier`), never both. Built as a string, so the fold is one piece of markup.

// The points, oldest first, and the end point, from lines and the game's present score.
function gameHistorySteps(lines, now) {
  const points = (Array.isArray(lines) ? lines : [])
    .filter(line => line && typeof line.index === 'number'
      && historyDayNumber(line.measured) !== null)
    .map(line => ({
      day: historyDayNumber(line.measured), measured: line.measured, index: line.index,
      version: String(line.weights_version || ''), refused: line.viable === false,
      n: typeof line.n === 'number' ? line.n : null,
    }))
    // Two on one day: the ledger's order where the lines carry it, else the older version
    // first (a visitor's lines carry no number and come newest first).
    .sort((a, b) => a.day - b.day
      || (a.n !== null && b.n !== null ? a.n - b.n
        : gameHistoryVersionNumber(a.version) - gameHistoryVersionNumber(b.version)));
  const end = now && now.status === 'current' && typeof now.index === 'number'
    && historyDayNumber(now.measured) !== null
    ? { day: historyDayNumber(now.measured), measured: now.measured, index: now.index,
        version: String(now.version || ''), refused: now.viable === false }
    : null;
  return { points, end };
}

function gameHistoryVersionNumber(version) {
  const found = /^v(\d+)/.exec(String(version || ''));
  return found ? Number(found[1]) : 0;
}

// `v12` as a reader says it: *version 12*.
function gameHistoryVersion(version) {
  const found = /^v(\d+)$/.exec(String(version || ''));
  return found ? 'version ' + found[1] : String(version || '');
}

function gameHistoryTag(tag, attrs, inner) {
  const said = Object.entries(attrs).map(([key, value]) => ` ${key}="${esc(value)}"`).join('');
  return inner === undefined ? `<${tag}${said}/>` : `<${tag}${said}>${inner}</${tag}>`;
}

// The chart as SVG markup, or '' with nothing to draw. `width` is the viewBox's, chosen by
// the caller for the screen it is drawn on.
function gameHistorySvg(steps, width) {
  const all = steps.points.concat(steps.end ? [steps.end] : []);
  if (!all.length) return '';
  const W = Math.max(280, Math.round(width || 480)), H = 180;
  const margin = { l: 34, r: 60, t: 34, b: 24 };
  const first = all[0].day;
  const last = Math.max(first + 1, ...all.map(point => point.day));
  const x = day => margin.l + (day - first) / (last - first) * (W - margin.l - margin.r);
  const y = historyScale(H, margin);
  const parts = [];
  const text = (tx, ty, words, attrs) => parts.push(gameHistoryTag('text', {
    x: tx.toFixed(1), y: ty.toFixed(1), 'font-size': 11, fill: 'var(--muted)',
    class: 'historyFigures', ...(attrs || {}) }, esc(words)));
  HISTORY_TICKS.forEach(tick => {
    parts.push(gameHistoryTag('line', { x1: margin.l, x2: W - margin.r,
      y1: y(tick).toFixed(1), y2: y(tick).toFixed(1), stroke: 'var(--line)', 'stroke-width': 1 }));
    text(margin.l - 6, y(tick) + 4, tick.toFixed(1), { 'text-anchor': 'end' });
  });
  text(x(first), H - 6, historyDay(all[0].measured), { 'text-anchor': 'start' });
  if (all[all.length - 1].day > first) {
    text(x(last), H - 6, historyDay(all[all.length - 1].measured), { 'text-anchor': 'end' });
  }
  // Each change of version: a dashed rule and the version's words, and no riser across it.
  // Words that would run into the previous ones go on the upper row.
  let labelled = -Infinity, row = 0;
  all.forEach((point, at) => {
    if (at && point.version === all[at - 1].version) return;
    const px = x(point.day);
    if (at) {
      parts.push(gameHistoryTag('line', { x1: px.toFixed(1), x2: px.toFixed(1),
        y1: 6, y2: H - margin.b,
        stroke: 'var(--muted)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
    }
    if (!point.version) return;
    row = px - labelled < 66 ? 1 - row : 0;
    text(px + (at ? 4 : 0), margin.t - 18 + row * 11, gameHistoryVersion(point.version),
      { 'text-anchor': 'start' });
    labelled = px;
  });
  // The steps: each value holds from its day to the next one's, rising to it only where
  // the next is under the same version.
  let d = '';
  all.forEach((point, at) => {
    const px = x(point.day).toFixed(1), py = y(point.index).toFixed(1);
    if (!at) { d = `M${px} ${py}`; return; }
    d += ` H${px}` + (point.version === all[at - 1].version ? ` V${py}` : ` M${px} ${py}`);
  });
  parts.push(gameHistoryTag('path', { d, fill: 'none', stroke: 'var(--pick-1)',
    'stroke-width': 2, 'stroke-linejoin': 'round' }));
  // A dot where the score changed (a first value, a new number, a new version), and on every
  // refused one and the present score; a re-measurement that stored the same number is the
  // step running on, as in the library page's day by day.
  all.forEach((point, at) => {
    const before = all[at - 1];
    if (before && point !== steps.end && !point.refused && point.index === before.index
      && point.version === before.version) return;
    const hollow = point.refused;
    const said = `${historyFigure(point.index)}, measured ${historyDay(point.measured, true)}${
      point.version ? ', ' + gameHistoryVersion(point.version) : ''}${
      hollow ? ', refused' : ''}${point === steps.end ? ', the present score' : ''}`;
    parts.push(gameHistoryTag('circle', { cx: x(point.day).toFixed(1),
      cy: y(point.index).toFixed(1), r: 4, fill: hollow ? 'var(--panel)' : 'var(--pick-1)',
      stroke: hollow ? 'var(--pick-1)' : 'var(--panel)', 'stroke-width': 2 },
    `<title>${esc(said)}</title>`));
  });
  const tail = all[all.length - 1];
  text(x(tail.day) + 8, y(tail.index) + 4,
    historyFigure(tail.index) + (tail === steps.end ? ' now' : ''),
    { fill: 'var(--ink)', 'font-weight': 600 });
  return gameHistoryTag('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
    class: 'historySvg', 'aria-label': "This game's stored scores over time" }, parts.join(''));
}
