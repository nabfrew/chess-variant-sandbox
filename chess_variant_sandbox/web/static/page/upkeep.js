// Keeping every score current, and the two tags that say whether one is.
//
// The panel is ``chess-variant-sandbox measure --check`` with buttons: the library could already say
// that forty scores were out of date -- every card carries the reason -- and the only
// way to act on it was a terminal.

// ------------------------------------------------------------- the scores as a whole

// `chess-variant-sandbox measure --check`, and then the thing that fixes what it finds. The library
// could already *say* that forty scores were stale -- every card carries the reason --
// and there was nowhere to act on it: the only way to re-take a measurement was a
// terminal, and for a committed ruleset the game page offered nothing at all.
//
// The counts are kept apart on purpose. "Never measured" and "measured under other
// code" are different news, and a re-score and a replay are three seconds and three
// minutes, so a panel that added them into one number would be hiding what the button
// is about to cost.

// What each repair does, in the words a row of this table needs. One per answer
// `Staleness.needs` gives; a row with no answer at all is a game nobody has measured,
// and playing it is what that costs.
//
// `refit` (Q80c, `docs/reference/refit-repair.md`) is the weakest of the four: the games
// and the scoring both stand, and only the objective the roster's fitted rungs play was
// accepted on a build other than this one, so it is checked here before anything else
// runs. Left unnamed it fell to the fallback below and read as a replay, which is the one
// thing it is not -- it costs no tournament unless the check itself changes the player.
const REPAIRS = {
  refit: 'checking its fit',
  rescore: 'scoring again',
  reanalyse: 'analysing and scoring again',
  replay: 'playing again',
};

// How many current scores rest on how many runs, fewest runs first. Only the current
// ones: a stale row's stored count is what it *had*, and the repair may change it.
function countRuns(games) {
  const tally = new Map();
  for (const game of games || []) {
    if (game.state !== 'current' || !game.runs) continue;
    tally.set(game.runs, (tally.get(game.runs) || 0) + 1);
  }
  return [...tally.entries()].sort((a, b) => a[0] - b[0]);
}

function scoringHtml(s) {
  if (!s || !s.games) return '';
  const behind = s.stale + s.absent;
  const sweep = s.sweep;
  const running = sweep && sweep.state === 'running';

  // Three repairs, not two. A `reanalyse` reads the games already on disk, like a
  // re-score, and runs the analysis pass over them before scoring -- so counting it with
  // the replays overstated what the button is about to cost. Named only when there is
  // one, because the rarest of the three is noise as a zero.
  const middle = s.reanalyse
    ? `${s.reanalyse} need the pass that reads those games to run again first; `
    : '';
  const cost = behind
    ? `<span class="muted small">${s.absent} have never been measured; ${s.stale} were
       measured by code that has since changed. ${s.rescore} of them only need scoring
       again from games already on disk, which is seconds each; ${middle}${s.replay} have
       to be played again, which is minutes each.</span>`
    : `<span class="muted small">Every one was played by this build against the same
       opponents and scored by the same rules.</span>`;

  // How many runs the up-to-date scores rest on. A run is one tournament from one seed
  // and a game's score is the mean of its runs, so a library half on one run and half on
  // four is a fact about the corpus rather than about any one card.
  const spread = countRuns(s.games);
  const rested = spread.length
    ? `<div class="muted small" style="margin-top:6px">Of the up-to-date ones, ${
        spread.map(([n, of]) => `${of} rest${of === 1 ? 's' : ''} on ${
          countWord(n)} run${n === 1 ? '' : 's'}`).join(' and ')
      }. A score is the mean of its runs${info('index')}.</div>`
    : '';

  // The core is behind its sources: it does not fail, it quietly plays the previous
  // build, so `measure` refuses and so does this. The buttons stay *there* and go grey
  // with the reason on them -- hiding them left this state as a paragraph of shell,
  // which is the one screen where a person can see forty stale scores and do nothing.
  //
  // The rebuild cannot be a button. It replaces the extension this process has loaded,
  // and on Windows that file is locked for as long as the server is running -- so the
  // honest instruction is to stop the server, rebuild, start it again, and it is spelt
  // out here rather than left as a command with no explanation of why.
  const build = s.build
    ? `<div class="verdict out" style="margin-top:10px">
         <div>Measuring is held, because the game engine has changed since it was last
           compiled.</div>
         <div class="small" style="font-weight:400; margin-top:5px">${esc(s.build)}.
           Anything measured now would describe the old build rather than this code, so
           the buttons are off until it is compiled again &mdash; stop this server, run
           that, and start it again.</div>
       </div>`
    : '';

  const held = s.build ? ` disabled title="${esc(s.build)}"` : '';
  const actions = running
    ? `<div class="row" style="margin-top:10px">
         <button id="stopSweep">stop after this game</button>
         <span class="muted small">It finishes the game it is on, because a tournament
         stopped half way through leaves something that looks measured and is not.</span>
       </div>`
    : `<div class="row" style="margin-top:10px">
        ${behind ? `<button class="primary" id="sweepStale"${held}>measure the ${behind}
          that need it</button>` : ''}
        <button id="sweepAll"${held} title="play every game again from scratch, whatever
          is on disk">measure all ${s.games.length} again</button>
        <span class="muted small">${s.build
          ? 'Compile the engine again and they come back.'
          : `It measures one game at a time in the background, so the page stays usable
             and you can leave it.`}</span>
      </div>`;

  return `<details class="panel" style="margin-bottom:14px" ${running ? 'open' : ''}>
    <summary>
      <b>${s.current} of the ${s.games.length} kept games have an up-to-date score</b>
      ${running ? `<span class="tag stale">measuring ${sweep.done + 1} of ${sweep.total}</span>` : ''}
    </summary>
    <p class="muted small" style="margin:10px 0 0">
      A score${info('index')} comes from a few hundred games the computer plays against
      itself. It depends on the rules, the opponents${info('roster')}, the scoring
      rules${info('weights')} and the code that ran both &mdash; so when any of those
      change, the old number stops describing this program: it is shown marked
      provisional${info('provisionalScore')} and compared with nothing until it is taken
      again${info('stale')}. This is where it is taken again. Currently
      ${(s.roster || []).length} opponents, ${s.gamesPerPairing} games per pairing,
      scoring rules ${esc(s.weights || '')}. Games nobody has kept yet${info('candidate')}
      are not counted here; measure one of those from its own page.
    </p>
    <div style="margin-top:8px">${cost}</div>
    ${rested}
    ${refresherHtml(s.refresher)}
    ${volatilityHtml(s.volatility)}
    ${build}
    ${actions}
    ${sweepHtml(sweep)}
    ${behind ? `<div class="cells" style="margin-top:10px"><table>
      <tr><th>game</th><th>needs</th><th>why</th></tr>
      ${s.games.filter(g => g.state !== 'current').map(g => `<tr>
        <td>${esc(g.name)}</td>
        <td>${esc(REPAIRS[g.needs] || 'playing again')}</td>
        <td class="muted">${esc(g.reason)}</td></tr>`).join('')}
    </table></div>` : ''}
  </details>`;
}

// Q705. The refresher (`chess-variant-sandbox refresh --library --watch`), as the server
// words its status file (`web/refreshing.py`): what it is doing now and on which game, how
// long ago it last wrote, what is owed, and every game it refused with why. With no file
// the server says it is not running; the page decides nothing and computes no age.
function refresherHtml(r) {
  if (!r || !r.said) return '';
  const refused = r.refused || [];
  return `<div class="refresher small" style="margin-top:10px">
      <b>The refresher</b>
      <ul class="rules small">${r.said.map(line => `<li>${esc(line)}</li>`).join('')}</ul>
      ${refused.length ? `<div class="cells"><table>
        <tr><th>refused game</th><th>repair</th><th>why</th></tr>
        ${refused.map(one => `<tr><td>${esc(one.name)}</td><td>${esc(one.needs || '')}</td>
          <td class="muted">${esc(one.why)}${one.reason ? `: ${esc(one.reason)}` : ''}</td>
          </tr>`).join('')}
      </table></div>` : ''}
    </div>`;
}

// EV6 (Q690). How much each kept game's stored score has moved across its revisions at
// one weights version, beside its interval, in the server's words (`history.volatility`).
// No cut-off and no colour: the reader compares the two numbers; the page computes none.
function volatilityHtml(rows) {
  if (!rows || !rows.length) return '';
  return `<details class="small" style="margin-top:10px">
      <summary><b>How much scores have moved</b></summary>
      <p class="muted">Across each game's last few re-measurements under the current
        scoring rules, beside the interval of the score it has now. This changes no score
        and no trust level.</p>
      <div class="cells"><table>
        <tr><th>game</th><th>how much it has moved</th></tr>
        ${rows.map(row => `<tr><td>${esc(row.name)}</td>
          <td class="muted">${esc(row.said)}</td></tr>`).join('')}
      </table></div>
    </details>`;
}

// What the sweep has actually done, which is the half a progress bar cannot carry: a
// game that failed is not a game that is slow, and a sweep that reported "finished"
// over three silent failures would be worse than no sweep at all.
function sweepHtml(sweep) {
  if (!sweep) return '';
  const state = { running: 'running', finished: 'finished', stopped: 'stopped' }[sweep.state]
    || sweep.state;
  const now = sweep.current
    ? ` &mdash; on <b>${esc(sweep.current.name)}</b> (${esc(sweep.current.needs)})`
    : '';
  const failed = sweep.items.filter(i => i.state === 'failed');
  const done = sweep.items.filter(i => i.state === 'finished');
  // The number a finished item shows, with the runs it rests on in the pill's own
  // words. Both are the item's (`keeping.SweepItem`), taken from the one measurement
  // together, and an item with no count carries no number either -- so a game whose
  // runs could not be scored reads *no score*, as its card does.
  const scoredAs = i => {
    const count = runsShort(i.seeds);
    return i.index === null || i.index === undefined || !count
      ? 'no score'
      : `${fmt(i.index)} <span class="muted">${esc(count)}</span>`;
  };
  return `<div class="small" style="margin-top:10px">
      The sweep is ${esc(state)}, with ${sweep.done} of ${sweep.total} done in
      ${Math.round(sweep.seconds || 0)}s${now}.
    </div>
    ${done.length ? `<div class="muted small" style="margin-top:4px">${done.slice(-6).map(i =>
      `${esc(i.name)} ${scoredAs(i)}${
        i.viable === false ? ' gated' : ''}`).join(' &middot; ')}</div>` : ''}
    ${failed.length ? `<div class="error" style="margin-top:6px">${failed.map(i =>
      `${esc(i.name)}: ${esc(i.error)}`).join('<br>')}</div>` : ''}`;
}

function wireScoring() {
  const stale = document.getElementById('sweepStale');
  const all = document.getElementById('sweepAll');
  const stop = document.getElementById('stopSweep');
  const begin = (scope, button) => {
    button.disabled = true;
    api('/api/scoring', { scope }).then(() => refreshScoring()).catch(error => {
      // The panel is at the top of a long library, so a refusal shown at the top of the
      // page and a refusal shown here are the same place only before anybody scrolls.
      button.disabled = false; fail(error, button);
    });
  };
  if (stale) stale.onclick = () => begin('stale', stale);
  // The only irreversible-feeling one: it replays games that nothing has invalidated,
  // and on a corpus this size that is hours. Asked plainly rather than hidden.
  if (all) all.onclick = () => {
    const total = (SCORING && SCORING.games.length) || 0;
    if (!confirm(`Replay all ${total} games on the corpus roster? Every score is taken`
      + ` again from scratch, which is minutes each. Nothing else is lost if you stop.`)) return;
    begin('all', all);
  };
  if (stop) stop.onclick = () => {
    stop.disabled = true;
    api('/api/scoring/stop', {}).then(() => refreshScoring()).catch(fail);
  };
}

// Repaint the panel alone, so a sweep that takes an hour does not keep rebuilding the
// library underneath somebody reading it. The full re-read happens once, when the sweep
// ends and the cards have new numbers on them.
//
// Left exactly as open or shut as the reader had it: this repaints every two seconds
// while a sweep runs, and a panel that sprang open under somebody's cursor twice a
// minute would be worse than showing no progress at all.
function paintScoring(scoring) {
  const box = document.getElementById('scoring');
  if (!box) return false;
  const was = box.querySelector('details');
  const open = was ? was.open : null;
  box.innerHTML = scoringHtml(scoring);
  const now = box.querySelector('details');
  if (now && open !== null) now.open = open;
  wireScoring();
  return true;
}

function refreshScoring() {
  return api('/api/scoring').then(scoring => {
    SCORING = scoring;
    if (!paintScoring(scoring)) return scoring;
    if (scoring.sweep && scoring.sweep.state === 'running') pollSweep();
    return scoring;
  });
}

function pollSweep() {
  if (SWEEP_TIMER) return;
  SWEEP_TIMER = setInterval(() => {
    api('/api/scoring').then(scoring => {
      SCORING = scoring;
      const running = scoring.sweep && scoring.sweep.state === 'running';
      if (!paintScoring(scoring)) { clearInterval(SWEEP_TIMER); SWEEP_TIMER = null; return; }
      if (!running) {
        clearInterval(SWEEP_TIMER); SWEEP_TIMER = null;
        // Every card's score may have moved, and the cards are what the panel was for.
        showLibrary().catch(fail);
      }
    }).catch(() => { clearInterval(SWEEP_TIMER); SWEEP_TIMER = null; });
  }, 2000);
}

// Three states, not two. "Unscored" and "stale" both arrive here with no index, and
// they are different news with different repairs: nobody has measured this, versus
// somebody did and not under the code you are running. Showing the second as the first
// hides the fact that a number exists and is being withheld, which is the half of the
// original bug a person would have to go looking for.
function scoreTag(game) {
  const stale = game.staleness && !game.staleness.current ? game.staleness : null;
  // A current official number always wins. Where there is none, a number measured on
  // somebody's own machine and checked by the server is drawn under its own words
  // (`provisionalTag`), never as the score itself.
  const official = !stale && game.index !== null && game.index !== undefined;
  if (!official && game.provisional) return provisionalTag(game.provisional);
  // EV2: a stale measurement with a number is the game's provisional estimate, drawn
  // with its number and the mark, never bare (`docs/plan-evolving-score.md` §5.1). The
  // run count stays inside the pill, as on a current one.
  if (stale && isProvisional(game) && game.index !== null && game.index !== undefined) {
    const count = runCount(game.seeds)
      ? ` <span class="muted">${esc(runsShort(game.seeds))}</span>` : '';
    const why = provisionalSentence(game) + (ACCOUNTS ? '' : ' ' + stale.summary);
    return `<span class="tag stale provisional" title="${esc(why)}">${fmt(game.index)}${
      count} provisional</span>${info('provisionalScore')}`;
  }
  if (stale) {
    // The maintainer's summary names the repair (*replay 1 of 4 runs*); the site says
    // what changed and what happens next (B9).
    const why = ACCOUNTS
      ? 'The computer players or the scoring changed since it was measured; it will be'
        + ' measured again.'
      : 'The score is out of date, and it has no number to show until the game is'
        + ' measured again. ' + stale.summary;
    return `<span class="tag stale" title="${esc(why)}">out of date</span>`;
  }
  const runs = runCount(game.seeds);
  if (game.index === null || game.index === undefined) {
    // Measured, and still without a number: too few of its runs could be read for a
    // mean to be honest. "Nobody has measured this" and "it was measured and cannot be
    // scored" are different news, and the second used to read as the first.
    if (runs) {
      const read = game.totalsUsed || 0;
      const why = ACCOUNTS
        ? `Too few of its ${countWord(runs)} runs could be scored, so it has no score
          rather than a low one.`
        : `It was measured over ${runsShort(game.seeds)}, and ${
          read === 0 ? 'none of them' : 'only ' + countWord(read) + ' of them'} could be
          scored -- the opponents could not read the games well enough. A mean needs three
          of four, so it has no number rather than a low one.`;
      return `<span class="tag" title="${esc(why.replace(/\s+/g, ' '))}">no score</span>`;
    }
    // A design run's own number, where there is one. Its own key and its own label,
    // because it is taken on a smaller roster and is not comparable with the corpus --
    // and its run count inside the pill, as the corpus number has. An archive entry
    // written before the archive kept its seeds has no count, and so no number either.
    if (game.designIndex !== null && game.designIndex !== undefined) {
      const by = `scored by the design run that invented it (${game.designRun})`;
      const count = runsShort(game.designSeeds);
      if (!count) {
        const why = by + ', which did not record how many runs the number rests on,'
          + ' so it is not shown';
        return `<span class="tag" title="${esc(why)}">no score</span>`;
      }
      // EV2: a design number under an earlier revision of the scorer is provisional too,
      // and says so in the pill, as a library number does.
      const later = game.designStatus === 'provisional';
      const why = by + `, as the ${runsPhrase(game.designSeeds)}, against fewer`
        + ' opponents -- not comparable with the rest'
        + (later ? '. ' + provisionalSentence({
          status: 'provisional', revision: game.designRevision }) : '');
      const pill = `${fmt(game.designIndex)} rough <span class="muted">${esc(count)}</span>${
        later ? ' provisional' : ''}`;
      return `<span class="tag" title="${esc(why)}">${pill}</span>`;
    }
    // A design run played it and could not score it: since C3 the loop archives such a
    // candidate with no index rather than a placeholder 0.0, so the run is named and its
    // number is null. Measured, so *no score*, as a corpus game whose runs could not be
    // scored reads -- never *not measured*, which is for a game nobody has played.
    if (game.designRun) {
      const count = runsShort(game.designSeeds);
      const why = `It was measured by the design run that invented it (${game.designRun})${
        count ? ', as the ' + runsPhrase(game.designSeeds) : ''}, and could not be scored:
        the opponents could not read its games well enough, or a measurement the score
        needs could not be taken. So it has no number rather than a low one.`;
      return `<span class="tag" title="${esc(why.replace(/\s+/g, ' '))}">no score</span>`;
    }
    return `<span class="tag">not measured</span>`;
  }
  const cls = game.viable === false ? 'tag gated' : 'tag score';
  // The run count is part of the number and never a detail beside it, so it is inside
  // the pill rather than in the hover -- a phone has no hover.
  const count = runs ? ` <span class="muted">${esc(runsShort(game.seeds))}</span>` : '';
  const label = game.viable === false
    ? `${fmt(game.index)} ruled out${count}` : `${fmt(game.index)}${count}`;
  const said = (game.viable === false
    ? 'scored, then ruled out because something about it does not work as a game'
    : 'the score out of 1, from a few hundred games')
    + (runs ? `, as the ${runsPhrase(game.seeds)}` : '')
    + (ACCOUNTS ? `, under ${rulesVersion(game.weights)}` : ` (scoring rules ${game.weights})`);
  const split = splitSentences(game.runVerdict, game.runs).join(' ');
  return `<span class="${cls}" title="${esc(said + (split ? '. ' + split : '.'))}">${
    label}</span>`;
}

// A score measured on somebody's own machine, which the server has checked and has not
// measured itself (`docs/plan-client-compute.md` §5, Q312). Official means the server
// played the games, and only an official score ranks a game or earns it a badge, so this
// one never borrows the score's pill: it says where it was measured and how far it was
// checked, in the two labels §5 names. `not yet confirmed` reaches only its designer --
// the server sends it to nobody else -- and `spot-checked` reaches everybody who can see
// the game, which is the user's answer A, kept after T2 showed a few forged games can
// move a score a long way. The label is what tells a reader that.
function provisionalWords(p) {
  const whose = p.yours ? 'your' : "its designer's";
  if (p.state === 'spot-checked') {
    const n = p.sample ? countWord(p.sample) + ' of its games' : 'a sample of its games';
    // B12: §3.2's definition of *spot-checked*, and *site* for *server*.
    return {
      short: 'spot-checked',
      label: `measured on ${whose} computer, spot-checked by the site`,
      why: `Spot-checked: the site replayed ${n} measured on ${whose} computer, and they`
        + ' came out exactly the same. A spot check catches most tampering but cannot'
        + ' rule it out, so this is not an official score: only a score the site measured'
        + ' itself ranks a game or earns it a badge.',
    };
  }
  return {
    short: 'not yet confirmed',
    label: 'measured on your computer, not yet confirmed',
    why: 'The site checked that every game follows the rules and scored them itself,'
      + ' and has not replayed any yet. Only you can see this number. It ranks nothing'
      + ' and earns no badge: only a score the site measured itself does.',
  };
}

function provisionalTag(p) {
  const words = provisionalWords(p);
  const count = runsShort(p.seeds);
  const runs = count ? ` <span class="muted">${esc(count)}</span>` : '';
  const scored = p.index !== null && p.index !== undefined;
  const number = !scored ? 'no score'
    : p.viable === false ? `${fmt(p.index)} ruled out,` : fmt(p.index);
  const said = words.label.replace(/^./, c => c.toUpperCase())
    + (count ? `, as the ${runsPhrase(p.seeds)}` : '')
    + (p.weights ? ` (scoring rules ${p.weights})` : '') + '. ' + words.why;
  const cls = p.viable === false ? 'tag gated provisional' : 'tag provisional';
  return `<span class="${cls}" title="${esc(said)}">${number} ${esc(words.short)}${runs}</span>`;
}

// The cheap gate's verdict, which answers a different question from the index and is
// the one somebody about to spend twenty minutes actually needs. "scored badly" and
// "cannot be won" are not the same news, and the score tag can only ever say the first
// -- it reads the index, so it is absent for anything unscored and it never carries a
// reason. `rps-hierarchy-chess` fails this gate by name and the library said nothing.
function playableTag(game) {
  if (game.playable !== false) return '';
  const why = (game.playableReasons || []).join('; ');
  return `<span class="tag gated" title="${esc(why)}">ruled out</span>`;
}

// The positive one, unlike the tag above, because neither answer is rare here and only
// one of them is news: an outside engine has a counterpart for these rules, which is an
// affordance rather than a warning. Half the library has no counterpart at all and
// saying so on twenty-five cards would be saying nothing on any of them.
//
// It says the rules translate and stops there, which is the whole of what the listing
// knows -- `library._fairy` explains why a card cannot ask the second question, and the
// game's own page is where an installed engine either agrees or is not offered. The
// tag's *text* says so too. It used to read "Fairy-Stockfish" bare, which a reader took
// as "plays it" -- and then opened `sardines`, tagged, to a picker with no such
// opponent, because the 2021 release cannot promote a custom pawn. Both were right and
// the tag had not said which question it was answering.
function fairyTag(game) {
  // Not on the site (B14): *translates* means nothing to a visitor, and a game's page
  // says whether Fairy-Stockfish plays it.
  if (ACCOUNTS || game.fairyTranslates !== true) return '';
  const why = 'These rules can be written out as a Fairy-Stockfish variant, so an engine'
    + ' somebody else wrote may be able to play them. That is all this tag says. Whether'
    + ' the build installed on this machine plays the same game is checked when you open'
    + ' the game, and its page says what it found.';
  return `<span class="tag" title="${esc(why)}">translates to Fairy-Stockfish</span>`;
}
