// ---------------------------------------------- measuring it on this computer (B5)
//
// `web/page/measure.py` says what this is. In one line: the browser plays and scores the
// draft with the package's own core and Python, on up to four workers, and this code only
// moves their text. It never reads a unit's request or reply. On the site, for a signed-in
// designer, it then sends the gathering half's upload -- a string it never parses -- and
// shows the server's own number and state in place of its own (B10).

let HERE_MANIFEST,    // `/browser/manifest.json`: undefined until asked, null where there is none
    HERE = null,      // the measurement in hand, running or finished
    HERE_CHOSEN = 0,  // the worker count picked, kept across redraws
    HERE_TICK = null, // redraws the clock while a measurement runs
    HERE_ASKED = false; // the manifest has answered, a build or none (UP7, `reachAsked`)

// Where the artefact and the worker are served (B4). `let` so a test can point them at a
// directory of its own.
let HERE_BASE = '/browser/',
    HERE_WORKER = '/static/measure-worker.js';

// Four workers at most (§8 B5), and one core left for everything else.
const HERE_MOST = 4;

// How often a submission's state is asked while the server walks it, in ms. `let` so a
// test need not wait.
let HERE_POLL = 2000;

// What each step is, in words. The keys are the gathering half's step names.
const HERE_STEPS = {
  loading: 'loading the engine into this browser',
  tournament: 'playing the tournament',
  reading: 'reading every game back',
  gathering: 'putting the pieces together',
};

function hereWorkerCap() {
  const cores = Math.floor(Number(navigator.hardwareConcurrency) || 1);
  return Math.max(1, Math.min(HERE_MOST, cores - 1));
}

function hereUrl(path) {
  return new URL(path, new URL(HERE_BASE, location.href)).href;
}

// The manifest, asked afresh and never cached (the server sends it `no-store` as well), so
// a measurement never starts a wheel the server has replaced. Anything but a manifest this
// page can read is none, and none offers nothing.
async function hereManifest() {
  try {
    const response = await fetch(hereUrl('manifest.json'), { cache: 'no-store' });
    // 204 is the local application saying, quietly, that it has no build (B10).
    if (!response.ok || response.status === 204) return null;
    const manifest = await response.json();
    const readable = manifest && manifest.schema === 1
      && typeof manifest.runtime === 'string' && typeof manifest.wheel === 'string';
    return readable ? manifest : null;
  } catch (error) {
    return null;
  }
}

// Asked once a page, when the designer opens. Until it answers, nothing is offered.
function hereLookup() {
  if (HERE_MANIFEST !== undefined) return;
  HERE_MANIFEST = null;
  hereManifest().then(manifest => {
    HERE_MANIFEST = manifest;
    HERE_ASKED = true;
    redrawHere();
  });
}

// One of B3's workers, and a way to ask it something. The worker answers every message
// but `close` exactly once and in the order sent, so the answers are matched to the
// questions in order.
function hereWorker(manifest, gathers) {
  const worker = new Worker(HERE_WORKER);
  const waiting = [];
  const helper = {
    heap: 0,
    ask(message) {
      return new Promise((resolve, reject) => {
        waiting.push({ resolve, reject });
        worker.postMessage(message);
      });
    },
    end(why) {
      worker.terminate();
      while (waiting.length) waiting.shift().reject(workerError(why));
    },
  };
  worker.onmessage = event => {
    const answer = event.data || {};
    if (typeof answer.heap === 'number') helper.heap = Math.max(helper.heap, answer.heap);
    const next = waiting.shift();
    if (!next) return;
    if (answer.kind === 'error') next.reject(workerError(answer.message || 'a worker failed'));
    else next.resolve(answer);
  };
  worker.onerror = event => {
    if (event && event.preventDefault) event.preventDefault();
    helper.end((event && event.message) || 'a worker could not start');
  };
  const load = {
    kind: 'load',
    runtime: hereUrl(manifest.runtime + '/'),
    wheel: hereUrl(manifest.wheel),
  };
  if (gathers) load.gathers = true;
  helper.loaded = helper.ask(load);
  return helper;
}

// A worker's own message is Python's or the browser's (*a unit arrived before Pyodide was
// loaded*, a traceback's last line), written for whoever debugs the worker. It is marked
// where it is made, so neither the site's answer about a send (Q682) nor the view of a
// measurement that stopped (Q728) shows it: the run keeps it as `error`, the console has
// it, and the reader gets `hereFault`'s plain clause.
function workerError(message) {
  const error = new Error(message);
  error.fromWorker = true;
  return error;
}

// What a failed send says after its colon: a sentence the server or this page wrote, as
// it is; a worker's message or a fault in the page, as a plain clause, with the detail in
// the console.
function hereFault(error) {
  if (error && error.fromWorker) {
    try { console.error(error); } catch (ignored) { /* no console */ }
    return 'measuring in this browser ran into a fault; reloading the page and measuring'
      + ' again usually works';
  }
  const words = errorWords(error);
  return words === PAGE_FAULT
    ? 'something went wrong on this page; reloading it usually puts it right' : words;
}

// The whole measurement. The first worker gathers and plays; the others play.
async function hereMeasure(spec, count) {
  const run = {
    hash: DRAFT_HASH, name: spec.name || 'this game', workers: count,
    state: 'loading', step: 'loading', done: 0, total: 0, games: 0,
    started: performance.now(), seconds: 0, times: [], loads: [], heaps: [],
    pool: [], result: null, error: '', said: '',
  };
  HERE = run;
  hereClock(true);
  redrawHere();
  try {
    const manifest = await hereManifest();
    if (!manifest) throw new Error("Measuring in the browser isn't available right now.");
    run.pool = Array.from({ length: count }, (_, i) => hereWorker(manifest, i === 0));
    run.loads = (await Promise.all(run.pool.map(helper => helper.loaded)))
      .map(loaded => loaded.seconds);
    run.times.push(['loading', (performance.now() - run.started) / 1000]);
    run.state = 'running';
    let answer = await hereGather(run, { spec, state: null });
    while (!answer.result) {
      const replies = await hereDeal(run, answer);
      answer = await hereGather(run, { state: answer.state, replies });
    }
    run.result = answer.result;
    // The upload the site reads, as the gathering half wrote it; a wheel from before B10
    // writes none, and then there is nothing to send.
    run.upload = typeof answer.upload === 'string' ? answer.upload : '';
    run.state = 'done';
  } catch (error) {
    if (run.state !== 'stopped') {
      run.state = 'failed';
      run.error = error.message || String(error);
      run.said = hereFault(error);
    }
  } finally {
    run.seconds = (performance.now() - run.started) / 1000;
    run.heaps = run.pool.map(helper => helper.heap);
    run.pool.forEach(helper => helper.end('finished'));
    if (HERE === run) {
      hereClock(false);
      redrawHere();
    }
  }
  if (run.state === 'done' && run.upload && hereSends()) await hereSettle(run);
}

// Only the site has people, and only a signed-in designer's own game can be sent.
function hereSends() {
  return ACCOUNTS && !!VIEWER;
}

// Q1004, the user: a measurement the site could not keep was measured all the same, and
// refused only once it was sent. The site keeps a score from this computer for a game its
// designer has saved, and for nothing else (`provisional.check_owner`), so the page asks
// the draft's standing first. What it answers, for the game the standing is of:
//
//   'local'    the local application, which sends nothing and says so up front;
//   'signin'   a visitor, who saves nothing; measuring here is off until they sign in;
//   'unknown'  no standing yet, which the end of a measurement asks again;
//   'send'     this person's saved game: the score goes to the site when it ends;
//   'save'     rules nobody has saved: measuring is on, and the finished score is held
//              on the page, with a button that saves the game and then sends it;
//   'library'  rules the library already holds, which the site measures itself;
//   'theirs'   rules somebody else saved first, whose score only they can send.
//
// The last two cannot be made this person's by saving, so measuring here is off for them,
// with the sentence saying how to make a game of their own (`HERE_OFF`). The page decides
// nothing about the score: a held one is the same upload, sent as it came once the game is
// saved, and the site walks and scores every game of it as it does any other (B10).
function hereKeeping(s) {
  if (!ACCOUNTS) return 'local';
  if (!VIEWER) return 'signin';
  if (!s) return 'unknown';
  const p = s.publish;
  if (p) return p.mine ? 'send' : 'theirs';
  return s.saved ? 'library' : 'save';
}

// Why measuring here is off, and how to turn it on, for each answer that turns it off.
const HERE_OFF = {
  signin: 'Measuring on your computer is for signed-in designers: the site keeps a score'
    + ' only for a game you have saved. Sign in and save this game, and you can measure'
    + ' it here.',
  library: 'These rules are already a library game, which the site measures itself, so a'
    + ' score from your computer could not be kept. Change a rule to make a game of your'
    + ' own, save it, and you can measure that here.',
  theirs: 'Somebody else saved these rules first, and only the person who saved a game can'
    + ' send it a score, so one from your computer could not be kept. Change a rule to make'
    + ' a game of your own, save it, and you can measure that here.',
};

// The standing of the game a measurement was of: the designer's own when it is still the
// draft on screen, else asked for, since the rules may have changed while it ran.
async function hereStanding(hash) {
  if (typeof DRAFT_STANDING !== 'undefined' && DRAFT_STANDING
      && DRAFT_STANDING.hash === hash) return DRAFT_STANDING;
  try {
    return await api('/api/design/standing/' + hash);
  } catch (error) {
    return null;
  }
}

// A finished measurement on the site: sent when its game is this person's saved game, held
// on the page while the game is unsaved, and kept back with the reason when it never can be
// this person's. With no standing to read, it is sent, and the site says what it says.
async function hereSettle(run) {
  const keeping = hereKeeping(await hereStanding(run.result.hash));
  if (keeping === 'save') run.check = { state: 'unsaved' };
  else if (HERE_OFF[keeping]) run.check = { state: 'unkept', why: HERE_OFF[keeping] };
  else return hereSubmit(run);
  if (HERE === run) redrawHere();
}

// The draft's standing has just been asked again (`loadStanding`): a score held for want of
// a save goes to the site once the standing says the game is this person's saved game,
// whichever button saved it.
function hereStood(s) {
  const run = HERE;
  if (!run || run.state !== 'done' || !run.upload || !run.check
      || run.check.state !== 'unsaved') return null;
  if (!s || s.hash !== run.result.hash || hereKeeping(s) !== 'send') return null;
  return hereSubmit(run);
}

// The finished measurement to the site, and the job followed until the server has walked
// it and, after that, spot-checked it (Q503e: `spotCheck` queued or running means keep
// asking). The upload goes as the string it arrived as. A `stale-build` refusal means this
// page loaded an engine the server no longer runs, so the manifest is asked again and the
// next measurement loads the current one. The designer's *Where this game stands* is asked
// again as soon as the site has taken the score, because a score the site is checking is
// what opens *Publish* (Q742), and again when the checks end, which is when a press that
// waited goes through or is called off; the page reads the server's answer and decides
// nothing itself.
async function hereSubmit(run) {
  run.check = { state: 'sending' };
  if (HERE === run) redrawHere();
  const restand = () => {
    if (typeof loadStanding === 'function' && typeof DRAFT_HASH === 'string'
        && DRAFT_HASH === run.result.hash) loadStanding(DRAFT_HASH);
  };
  try {
    let answer = await api('/api/provisional', { game: run.result.hash, upload: run.upload });
    if (answer.state === 'queued' || answer.state === 'running') restand();
    while (answer.state === 'queued' || answer.state === 'running' || hereSpotting(answer)) {
      run.check = answer;
      if (HERE === run) redrawHere();
      await new Promise(resolve => setTimeout(resolve, HERE_POLL));
      answer = await api('/api/provisional/' + answer.job);
    }
    run.check = answer;
    if (answer.refusal && answer.refusal.name === 'stale-build') {
      HERE_MANIFEST = undefined;
      hereLookup();
    }
    restand();
  } catch (error) {
    run.check = { state: 'failed', error: hereFault(error) };
  }
  if (HERE === run) redrawHere();
}

// Whether the site has walked the score and is still to spot-check it.
function hereSpotting(answer) {
  const spot = answer && answer.spotCheck;
  return !!spot && (spot.state === 'queued' || spot.state === 'running');
}

// One step of the gathering half. Its answer is the one text here that is parsed: an
// envelope of unit names, counts and strings, whose strings go on to a worker unread.
async function hereGather(run, request) {
  run.step = 'gathering';
  redrawHere();
  const started = performance.now();
  const reply = await run.pool[0].ask({
    kind: 'gather', id: 'gather', request: JSON.stringify(request),
  });
  run.times.push(['gathering', (performance.now() - started) / 1000]);
  return JSON.parse(reply.text);
}

// A step's units, each to the next worker free, and each reply put at its unit's place.
async function hereDeal(run, answer) {
  const units = answer.units || [];
  const replies = {};
  run.step = answer.step;
  run.done = 0;
  run.total = units.length;
  redrawHere();
  const started = performance.now();
  let next = 0;
  await Promise.all(run.pool.map(async helper => {
    while (next < units.length) {
      const unit = units[next];
      next += 1;
      const reply = await helper.ask({ kind: 'unit', id: unit.name, request: unit.request });
      (replies[unit.phase] = replies[unit.phase] || [])[unit.part] = reply.text;
      run.done += 1;
      if (unit.plays) run.games += unit.games;
      redrawHere();
    }
  }));
  run.times.push([answer.step, (performance.now() - started) / 1000]);
  return replies;
}

function hereStop() {
  const run = HERE;
  if (!run || (run.state !== 'loading' && run.state !== 'running')) return;
  run.state = 'stopped';
  run.pool.forEach(helper => helper.end('stopped'));
}

function hereClock(on) {
  if (HERE_TICK) {
    clearInterval(HERE_TICK);
    HERE_TICK = null;
  }
  if (on) HERE_TICK = setInterval(redrawHere, 1000);
}

function hereSeconds(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  if (s < 60) return s.toFixed(1) + ' s';
  const minutes = Math.floor(s / 60);
  return minutes + ' min ' + Math.round(s - minutes * 60) + ' s';
}

// A worker is one of the computer's cores, which is the word a person has for it (D38).
function hereWorkers(n) {
  return n === 1 ? 'one core' : countWord(n) + ' cores';
}

function hereCores(n) {
  return n === 1 ? "one of this computer's cores"
    : `up to ${countWord(n)} of this computer's cores at once`;
}

function redrawHere() {
  const box = document.getElementById('dHere');
  if (!box) return;
  box.innerHTML = hereHtml(DRAFT_CHECK, box.dataset.in === 'panel');
  wireHere();
}

// Nothing at all where there is no artefact, or no game to measure, and nothing in hand.
// `inPanel` is the control drawn inside *Publish*, under the line asking for a score
// (Q742, `hereInPanel`): a framed step of that panel rather than a section of its own.
function hereHtml(check, inPanel) {
  const run = HERE;
  const offer = !!HERE_MANIFEST && !!check && check.valid && check.playable !== false;
  if (!run && !offer) return '';
  const busy = !!run && (run.state === 'loading' || run.state === 'running');
  const keeping = hereKeeping(check && typeof standingFor === 'function'
    ? standingFor(check) : null);
  return `<div class="${inPanel ? 'hereStep' : 'ruled'}">
      ${inPanel ? '<div class="u-fw-600">Measure it on this computer</div>'
        : '<h3 class="u-mt-0">Measure it on this computer</h3>'}
      ${run ? hereRunHtml(run) : inPanel ? `<div class="muted small">This computer plays the
        games and scores them; a game the size of chess takes a few minutes. Then the site
        checks the score.</div>` : `<div class="muted small">Your browser plays the games the
        server would, with the same engine, and scores them the same way, on ${
        hereCores(hereWorkerCap())}. A game the size of chess takes a few minutes. ${
        HERE_OFF[keeping] ? '' : hereSends() ? `When it finishes, the games go to the site,
        which checks every one and scores them itself; the number you keep is the site's.${
        keeping === 'save' ? ` This game is not saved yet: when the measurement ends, save
        it and the score is kept.` : ''}`
        : 'Nothing is sent anywhere and nothing is kept: leave this page and it is gone.'}</div>`}
      ${offer && !busy ? HERE_OFF[keeping] ? hereOffHtml(run, keeping) : hereOfferHtml(run) : ''}
    </div>`;
}

// The button, drawn and not working, with what turns it on (Q1004): measuring here is off
// wherever the site could not keep what it measured.
function hereOffHtml(run, keeping) {
  return `<div class="row u-mt-8">
      <button id="dHereGo" disabled title="${esc(HERE_OFF[keeping])}">${
        run ? 'measure it here again' : 'measure it here'}</button>
    </div>
    <div class="muted small u-mt-6" id="dHereOff">${esc(HERE_OFF[keeping])}</div>`;
}

function hereOfferHtml(run) {
  const most = hereWorkerCap();
  const chosen = Math.min(most, HERE_CHOSEN || most);
  const options = Array.from({ length: most }, (_, i) => i + 1).map(n =>
    `<option value="${n}"${n === chosen ? ' selected' : ''}>${n}</option>`).join('');
  return `<div class="row u-mt-8">
      <button id="dHereGo"${run ? '' : ' class="primary"'}>${
        run ? 'measure it here again' : 'measure it here'}</button>
      <label class="muted small">use <select id="dHereWorkers">${options}</select>
        cores</label>
    </div>`;
}

function hereRunHtml(run) {
  if (run.state === 'done') return hereResultHtml(run);
  const spent = hereSeconds((performance.now() - run.started) / 1000);
  if (run.state === 'failed') {
    return `<div class="error">It stopped before it finished: ${esc(run.said || run.error)}</div>`;
  }
  if (run.state === 'stopped') {
    return `<div class="muted small">Stopped after ${esc(hereSeconds(run.seconds))}.
      Nothing was kept.</div>`;
  }
  const share = run.total ? Math.round(100 * run.done / run.total) : 0;
  return `<div class="small"><b>${esc(HERE_STEPS[run.step] || run.step)}</b>${
      run.total && run.step !== 'gathering' ? ` &mdash; ${run.done} of ${run.total}` : ''}</div>
    <div class="bar u-m-6-0"><div class="${styleClass(`width:${share}%`)}"></div></div>
    <div class="muted small">${esc(spent)} so far, on ${esc(hereWorkers(run.workers))}${
      run.games ? ` &middot; ${run.games} games played` : ''}. You can keep editing; this
      measures ${esc(run.name)} as it stood when you pressed the button.</div>
    <div class="row u-mt-8"><button id="dHereStop">stop</button></div>`;
}

// The number at the four decimals B5's acceptance compares, under the label that says
// where it was made and that nothing has checked it.
function hereResultHtml(run) {
  const r = run.result;
  const moved = !DRAFT_CHECK || DRAFT_CHECK.hash !== r.hash;
  const components = r.components || [];
  const check = run.check || null;
  const kept = check && check.state === 'done' && check.result ? check.result : null;
  return `${kept ? hereKeptHtml(check, kept) : hereOwnHtml(run, check)}
    ${moved ? `<div class="muted small u-mt-6">This is ${esc(r.name)} as it
      stood when you pressed the button${ACCOUNTS ? '' : ` (<code>${esc(r.hash)}</code>)`}; the rules have changed
      since, so it is not a number for the draft on screen.</div>` : ''}
    ${(r.violations || []).length ? `<div class="verdict out">
        <div>It is ruled out${info('gated')}, not marked down.</div>
        <div class="small u-fw-400 u-mt-5">${
          r.violations.map(esc).join('<br>')}</div></div>` : ''}
    ${components.length ? `<div class="muted small u-mt-8">What it is made of${
      kept ? ', as your browser scored it' : ''}, each out of 1, the ones that count most
      first:</div>
      <ul class="rules small u-mt-4">${components.map(c => `<li><b>${
        esc(c.title)}</b> ${Number(c.score).toFixed(2)} &mdash; ${esc(c.what || '')}</li>`)
        .join('')}</ul>` : ''}
    <details class="u-mt-6"><summary class="muted small">how long it took</summary>
      <div class="muted small u-mt-4">${hereTimesHtml(run)}</div></details>`;
}

// The browser's own number, under the label that says where it was made and that nothing
// has checked it, and what the site is doing with it, if anything.
function hereOwnHtml(run, check) {
  const r = run.result;
  const scored = r.index !== null && r.index !== undefined;
  const unsent = `Nothing has checked it. It was measured in this browser, ${
    hereSends() && !run.upload
      ? 'by an older version of this page (reload and measure again), ' : ''}sent nowhere and
    kept nowhere, so it ranks nothing and earns nothing.${ACCOUNTS && !VIEWER
    ? ' Sign in, and a score measured here goes to the site to be checked.'
    : ' The number a game keeps is the one the site measures once it is saved.'}`;
  return `<div class="u-fs-26 u-fw-600 u-lh-1_1">${
      scored ? Number(r.index).toFixed(4) : '<span class="muted">no score</span>'}</div>
    <div class="muted small">measured on your computer${info('provisional')}, as one run of ${
      Number(r.games)} games${isNotMeasured(r) ? ', and not measured'
      : r.viable === false ? ', and ruled out' : ''} &middot; ${
      esc(rulesVersion(r.weights))}${info('weights')}</div>
    ${check ? hereCheckHtml(check, run)
      : `<div class="small u-mt-6">${unsent}</div>`}`;
}

// A score held on the page because its game is not saved (Q1004). While the draft on screen
// is still the game it measured, one press saves the game the way *save it* does, and the
// score goes to the site once the save lands (`hereStood`). Once the rules have changed,
// saving would save another game, so it says how to get back to one it can keep.
function hereUnsavedHtml(run) {
  const here = typeof DRAFT_HASH === 'string' && DRAFT_HASH === run.result.hash;
  if (!here) {
    return `<div class="small u-mt-6" id="dHereUnsaved">The site keeps a score only for a
      saved game, and these rules were never saved. They have changed since you pressed
      the button, so this score cannot be kept as things stand: put the rules back as they
      were and save them, or measure the draft on screen.</div>`;
  }
  return `<div class="small u-mt-6" id="dHereUnsaved">The site keeps a score only for a saved
      game, and this one is not saved yet. Save it, and these games go to the site to be
      checked, as they would have if it had been saved first; nothing is measured again.
      Leave this page without saving and the score is gone.</div>
    <div class="row u-mt-8"><button id="dHereSave" class="primary"
      >save it and keep the score</button></div>`;
}

// The site's answer while it has no number to show, in its own words.
function hereCheckHtml(check, run) {
  if (check.state === 'unsaved') return hereUnsavedHtml(run);
  if (check.state === 'unkept') {
    return `<div class="small u-mt-6" id="dHereUnkept">It was not sent: ${esc(check.why)}</div>`;
  }
  if (check.state === 'sending') {
    return '<div class="small u-mt-6">Sending the games to the site&hellip;</div>';
  }
  if (check.state === 'queued' || check.state === 'running') {
    return `<div class="small u-mt-6">The site is ${check.state === 'queued'
      ? 'about to check' : 'checking'} it: every game replayed to check it was legal, and
      scored again by the site. Nobody else sees any of it.</div>`;
  }
  const refusal = check.refusal || null;
  if (refusal && refusal.name === 'stale-build') {
    return `<div class="error u-mt-6">The site runs a newer engine than the
      one this page loaded, so it could not take this score. The page has fetched the
      current one: measure it here again.</div>`;
  }
  if (check.state === 'done' && check.superseded === 'mismatch') {
    return `<div class="error u-mt-6">The site replayed some of these games
      and one came out differently from what this computer sent, so it did not keep this
      score. Your account can't publish until the person who runs the site has looked at
      it.</div>`;
  }
  if (check.state === 'done') {
    return `<div class="small u-mt-6">The site checked it, and its number no
      longer stands: ${check.superseded === 'superseded'
      ? 'a newer score for this game replaced it'
      : 'the site has changed how it scores games since, so this score is out of date'}.
      </div>`;
  }
  const said = refusal ? refusal.sentence : check.error || 'the site did not take it';
  return `<div class="error u-mt-6">The site did not keep it: ${esc(said)}</div>
    <div class="row u-mt-8"><button id="dHereSend">send it again</button></div>`;
}

// The server's number in place of the browser's (B10), under the state and the label the
// game's own page gives it (`provisionalWords`).
function hereKeptHtml(check, p) {
  const words = provisionalWords(p);
  const scored = p.index !== null && p.index !== undefined;
  const took = typeof check.seconds === 'number' ? ` in ${hereSeconds(check.seconds)}` : '';
  return `<div class="u-fs-26 u-fw-600 u-lh-1_1">${
      scored ? Number(p.index).toFixed(4) : '<span class="muted">no score</span>'}</div>
    <div class="muted small">${esc(words.label)}${info('provisional')}${
      isNotMeasured(p) ? ', and not measured' : p.viable === false ? ', and ruled out' : ''
    } &middot; ${esc(runsPhrase(p.seeds))}${
      p.weights ? ' &middot; ' + esc(rulesVersion(p.weights)) : ''}${info('weights')}</div>
    <div class="small u-mt-6">Checked by the site${esc(took)}. ${
      esc(words.why)}</div>${hereSpotHtml(check)}`;
}

// Tier two's word, under the number while it is only `legal` (Q503e).
function hereSpotHtml(check) {
  const spot = check.spotCheck;
  if (!spot || (check.result && check.result.state === 'spot-checked')) return '';
  if (spot.state === 'queued' || spot.state === 'running') {
    return `<div class="small u-mt-6">The site is now replaying a sample of
      the games exactly, to spot-check it; that takes a few minutes.${
      typeof hereInPanel === 'function' && hereInPanel(DRAFT_STANDING)
        ? ' You can press Publish already: the game goes on trial by itself once the check has passed.'
        : ''}</div>`;
  }
  if (spot.state === 'stopped') {
    return `<div class="small u-mt-6">The site's spot check ran out of time
      on this game, so the score stays yours alone and cannot publish it yet.</div>`;
  }
  if (spot.state === 'failed') {
    return `<div class="small u-mt-6">The site could not finish the spot
      check; send the score again.</div>
      <div class="row u-mt-8"><button id="dHereSend">send it again</button></div>`;
  }
  return '';
}

function hereTimesHtml(run) {
  const spent = {};
  for (const [step, seconds] of run.times) spent[step] = (spent[step] || 0) + seconds;
  // In the order the steps run, with the gathers between them added up at the end.
  const steps = Object.keys(HERE_STEPS).filter(step => step in spent).map(step =>
    `${esc(HERE_STEPS[step])}, ${esc(hereSeconds(spent[step]))}`);
  const heap = Math.max(0, ...run.heaps);
  return `${esc(hereSeconds(run.seconds))} in all, on ${esc(hereWorkers(run.workers))}: ${
    steps.join('; ')}.${heap && !ACCOUNTS ? ` Each worker's engine used at most ${
    Math.round(heap / 1048576)} MiB.` : ''}`;
}

function wireHere() {
  const pick = document.getElementById('dHereWorkers');
  if (pick) pick.onchange = () => { HERE_CHOSEN = Number(pick.value) || 0; };
  const go = document.getElementById('dHereGo');
  if (go) go.onclick = () => {
    const most = hereWorkerCap();
    const count = Math.max(1, Math.min(most, Number(pick && pick.value) || most));
    hereMeasure(structuredClone(DRAFT), count);
  };
  const stop = document.getElementById('dHereStop');
  if (stop) stop.onclick = () => hereStop();
  const send = document.getElementById('dHereSend');
  if (send && HERE && HERE.upload) send.onclick = () => hereSubmit(HERE);
  // The editor's own save, name question and all; what the save says lands where *save it*
  // puts it, and the standing it asks for afterwards sends the score (`hereStood`).
  const save = document.getElementById('dHereSave');
  if (save && HERE && HERE.upload) save.onclick = () => {
    save.disabled = true;
    const saving = saveDraft();
    const again = () => { if (save.isConnected !== false) save.disabled = false; };
    if (saving && typeof saving.then === 'function') saving.then(again, again);
    else again();
  };
  // UP7: the profile's *Next* link asks for this control; it is reached once it stands.
  if (typeof reachAsked === 'function') reachAsked();
}
