// One game: what it is, how it scored, and what can be done with it.

// --------------------------------------------------------------------- one game

async function showGame(hash) {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  // The game first, so the roster reads it from the cache rather than fetching it again.
  const game = await detailFor(hash);
  // G3: the tab names the game, as the server's title for its address does.
  setTitle(gameTitle(game.name));
  const anchored = offered('GET', '/api/human/' + hash);
  const [seats, mine] = await Promise.all([
    roster(hash), anchored ? humanFor(hash) : null,
  ]);
  // The game page's half of the same answer the listing gives, for the header's link.
  noteAdmin(game.standing && game.standing.admin);
  // A game whose rules have left the corpus. The server says so; the page never works it
  // out from an absent spec, and the local application, which sends no standing at all,
  // draws every game the ordinary way.
  const retired = !!(game.standing && game.standing.retired);

  // The name sits above the two columns rather than at the top of the left one. Stacked
  // on a phone the play column comes first, so a name inside the left column read after
  // the whole play panel and the crumb above it was all the page said this game was.
  // The hash closes the line on a checkout only: it is the maintainer's name for these
  // rules, and on the site the address already carries it (Q576, F6).
  // *Make your own from this* sits under the name for the same reason. It was in the left
  // column, under the description, so from Q256 C6 (3b945e8f, the play column first on a
  // phone) it was some ten thousand pixels down a phone's page and read as gone (Q642).
  view.innerHTML = `
    <button class="crumb" id="back">&larr; all games</button>
    <h2 class="u-m-10-0-0">${esc(game.name)}${
      retired ? ' <span class="tag">retired</span>' : ''}${
      game.preview ? ' <span class="tag level preview">Preview</span>' : ''}</h2>
    ${previewHtml(game.preview)}
    <div class="row u-mt-6">
      <button id="fork" title="open a copy of these rules in the designer, under a new name; this game stays as it is"
        >make your own from this</button>
    </div>
    <div class="row muted small u-gap-8 u-mt-10">
      <span>${esc(game.board)} board</span><span>&middot;</span>
      <span>${game.pieceTypes} piece type${game.pieceTypes === 1 ? '' : 's'}</span>
      <span>&middot;</span>
      <span>complexity ${game.ruleMass}${info('ruleMass')}</span>
      ${ACCOUNTS ? '' : `<span>&middot;</span>
      <span title="its content hash, which names this exact set of rules" class="u-ff-mono">${esc(game.hash)}</span>`}
    </div>
    <p class="gameAbout">${esc(game.description)}</p>
    <div class="gameTop">
      <div class="gamePlay">
        ${carryOnHtml(game)}
        ${openingHtml(game)}
        <div class="panel">
          ${retired ? retiredBody(game) : `<h3 class="u-mt-0">Play</h3>
          ${playableWarning(game)}
          ${firstGameHtml(game)}
          ${sidePicker(game, seats, ratingTable(game))}
          ${fairyNote(game, seats)}
          ${modePicker(game)}
          <div class="row u-mt-10">
            <button class="primary" id="start">start a game</button>
            <button id="swap" title="take the other side">swap sides</button>
            <button id="watchIt" title="${esc(watchTitle())}">watch it play itself</button>
            ${offered('POST', '/api/opponent')
              ? `<button id="buildBot" title="an opponent that plays for what you tell it to"
                  >build an opponent</button>` : ''}
          </div>`}
        </div>
        <div id="people">${playPersonHtml(game)}</div>
        ${provenHtml(game)}
      </div>
      <div class="gameSide">
        ${standingHtml(game)}
        ${votingHtml(game)}
        <div class="panel"${ACCOUNTS ? ' id="leaderboard"' : ''}>
          ${ACCOUNTS
            ? `<h3 class="u-mt-0">Leaderboard${info('leaderboard')}</h3>
              <div id="strength">${leaderboardHtml(game)}</div>`
            : `<h3 class="u-mt-0">How hard each opponent is, here${info('elo')}</h3>
              <div id="strength">${strengthHtml(game)}</div>`}
          ${opponentLineHtml(game.standing)}
        </div>
        ${anchored ? yourRecordHtml(mine) : ''}
      </div>
    </div>
    <div class="gameRest">
      <div>
        ${sectionHtml('score', `How it scored${info('index')}`, `<div class="panel">
          ${exhibitHtml(game.exhibit)}${previewHtml(game.preview)}
          <div id="keepState">${measuringHtml(game)}</div>
          ${provisionalHtml(game)}
          ${scoreHtml(game.score, game.scoreGuide, game.band, game.trust)}
          ${wasHtml(game.history)}
        </div>`, true)}
        ${whereHtml(game)}
        ${verdictHtml(game.verdict)}
        ${linksHtml(game)}
        ${offered('POST', '/api/run') ? sectionHtml('designRun',
          `Let a model try to improve it${info('designRun')}`, `<div class="panel">
          <div class="muted small">
            It writes variations on this game, plays a few hundred games of each, scores
            them and keeps what survives &mdash; then starts again from the best. Each
            round costs a model call and a few minutes.
          </div>
          <textarea id="theme" placeholder="Write something to try, if you have one in mind. Left empty, the model works out what makes this game interesting and pushes on that."></textarea>
          <div class="row u-mt-8">
            <label class="muted small">rounds
              <input id="generations" type="number" min="1" max="40" value="4" class="u-w-72">
            </label>
            <button id="evolve">start</button>
          </div>
        </div>`) : ''}
      </div>
      <div>
        ${sectionHtml('pieces', 'The pieces',
          `<div class="panel pieces">${pieceBoardsHtml(game)}</div>`)}
        ${sectionHtml('rules', 'The rules', `<div class="panel">
          <ul class="rules">${(game.ruleLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
          <details class="u-mt-10">
            <summary class="muted small">where the pieces start</summary>
            <ul class="rules muted small u-mt-6">${
              (game.setupLines || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
          </details>
          ${reportHtml(game)}
        </div>`)}
        ${sectionHtml('kind', `What kind of game it is${info('kind')}`, `<div class="panel">
          <div class="axes">${(game.axes || []).map(a => `
            <div class="axis">
              <b>${esc(a.axis)}</b>
              <div class="asks muted small">${esc(a.asks)}</div>
              <div class="u-mt-3"><b class="u-fs-14">${esc(a.bin)}</b></div>
              <div class="muted small">${esc(a.means)}</div>
            </div>`).join('')}</div>
        </div>`)}
        ${betweenHtml(game)}
        ${sectionHtml('came', 'Where it came from',
          `<div class="panel">${provenanceHtml(game.provenance, game)}</div>`)}
      </div>
    </div>`;

  document.getElementById('back').onclick = () => { pending('Loading the library…'); go('#/'); };
  document.getElementById('fork').onclick = () => go('#/design/' + hash);
  const buildBot = document.getElementById('buildBot');
  if (buildBot) buildBot.onclick = () => go('#/opponent/' + hash);
  // A retired game has no play panel, so there are no three buttons to wire.
  if (!retired) {
    const start = document.getElementById('start');
    const first = document.getElementById('seat0'), second = document.getElementById('seat1');
    // The mode control exists only where the store does, and a game that counts is
    // something a player chooses rather than something they get by leaving a field out.
    const mode = document.getElementById('mode');
    const modeSays = document.getElementById('modeSays');
    // The button and the line under the menu follow every change to the three menus.
    // A player no game against is rated (the learned player, Q683) sets the game to
    // practice and holds the menu there while it is chosen; choosing another gives the
    // menu back at the mode this browser last started a game in.
    const nameStart = () => {
      const only = practiceOnlyIn([first.value, second.value]);
      if (mode && only) {
        mode.value = 'sandbox';
        mode.disabled = true;
      } else if (mode && mode.disabled) {
        mode.disabled = false;
        mode.value = lastMode();
      }
      start.textContent = startWords([first.value, second.value], mode ? mode.value : '');
      if (mode && modeSays) {
        modeSays.textContent = only ? practiceOnlySays(only) : MODE_SAYS[mode.value] || '';
      }
    };
    first.onchange = second.onchange = nameStart;
    if (mode) mode.onchange = nameStart;
    wireSeatCards();
    nameStart();
    start.onclick = () => {
      const seats = [first.value, second.value];
      const body = { hash, seats };
      // A first game here opens with the rules beside the board (K3); either way this
      // browser now has played here.
      if (firstSeat(game) && seats.includes('human')) OPEN_TAB = 'rules';
      notePlayed(hash);
      if (mode) {
        body.mode = mode.value;
        if (!mode.disabled) keepMode(mode.value);
      }
      pending('Starting the game…');
      // Under the play panel's own button row. A game page is long and this button is
      // usually the one thing on screen when it is pressed.
      api('/api/session', body).then(open_).catch(error => fail(error, 'start'));
    };
    document.getElementById('swap').onclick = () => {
      [first.value, second.value] = [second.value, first.value];
      first.dispatchEvent(new Event('change'));
      second.dispatchEvent(new Event('change'));
    };
    // Both seats to the same searching player, at the budget every watch on this page
    // uses (`WATCH_AGENT`). It was reachable already -- set both menus to a computer and
    // press start -- which is a thing you have to know rather than a thing you can see.
    // No mode goes with it: nobody plays a watch, so there is nothing to rate.
    wireFirstGame(game, hash);
    document.getElementById('watchIt').onclick = () => {
      pending('Starting the game…');
      api('/api/session', { hash, seats: [WATCH_AGENT, WATCH_AGENT] })
        .then(open_).catch(error => fail(error, 'watchIt'));
    };
  }
  wireCarryOn();
  wirePlayPerson(game, hash);
  wireLinks();
  wireBetween(game);
  wireDeleting(game, hash);
  wireReporting(game);
  const evolve = document.getElementById('evolve');
  if (evolve) evolve.onclick = () => {
    api('/api/run', {
      hash,
      theme: document.getElementById('theme').value,
      generations: Number(document.getElementById('generations').value) || 4,
    }).then(run => { RUN = run; go('#/run/' + run.id); }).catch(fail);
  };
  wirePlans();
  wireBadges(game);
  wireVoting(game);
  wireRating(hash);
  wireMeasuring(hash);
  wireKeeping(hash);
  wireNominating(game);
  wireRoundVote(game);
  const unrated = game.pickerRatings === null || game.pickerRatings === undefined;
  if (unrated && offered('GET', '/api/rating/' + hash)) {
    // Nothing stored, but a run may already be going for this ruleset in another tab.
    api('/api/rating/' + hash).then(state => {
      if (state.state === 'running') pollRating(hash);
    }).catch(() => {});
  }
}

// ------------------------------------------------------------- the start position
//
// K1 (`docs/archive/plan-understanding.md` §4.2 step 1): the page opens on the game, drawn. The
// opening position stands at the head of the play column before any game is started,
// with the marks the rules put on squares and the legend line naming them, so a goal
// square or a promotion band is seen before it is read about. The drawing is the
// server's `opening` (the card's `still`, its pieces' drawings put back) through the
// still board every preview uses; the page places nothing. The list of where each piece
// starts stays, folded under *The rules*, for a screen reader.
function openingHtml(game) {
  const opening = game && game.opening;
  if (!opening) return '';
  const preview = Object.assign({}, opening, { hash: game.hash });
  const kinds = new Set();
  const marks = squareMarks(preview, { zone: !zonesShown() });
  for (const square of Object.keys(marks)) {
    for (const mark of marks[square]) kinds.add(mark.kind);
  }
  boardKinds(preview.obstacles, preview.portals).forEach(kind => kinds.add(kind));
  // Schema 30 (Q999g): the effects the start position already carries.
  for (const [, bits] of effectPairs(preview.effects)) effectKinds(bits).forEach(kind => kinds.add(kind));
  // Q852: each wild piece's walk, drawn in order on the start position.
  if (walksDrawn(preview)) kinds.add('walk');
  const said = legendEntries(kinds, preview.wrap, '', preview.portals);
  // Never taller than most of a screen: a board taller than it is wide is held to the
  // width that leaves its foot in view.
  const tall = opening.height / Math.max(1, opening.width);
  return `<figure class="opening ${styleClass(`max-width:min(100%, calc(52vh / ${
    tall.toFixed(3)}))`)}" id="opening">
    <div role="img" aria-label="The start position. Where each piece starts is also listed under The rules.">${
      boardPreview(preview, null)}</div>
    <figcaption class="muted small">The start position${
      game.rounds ? ': both players move at once, every round'
      : (game.sides || [])[opening.toMove] ? `: ${esc(game.sides[opening.toMove])} moves first` : ''}.</figcaption>
    ${said.length ? `<div class="frame-legend small" data-opening="legend">on the board: ${
      said.join(' ')}</div>` : ''}${walkLeapsHtml(preview)}
  </figure>`;
}

// ------------------------------------------------------------- the page's sections
//
// V6 (`docs/plan-visual.md` §3.4): under the play panel and its companions, the rest of
// the game is a run of sections, each a heading that folds what is under it. At a desk
// they open; on a phone only the score starts open, so the page reads as a list of what
// there is to know rather than ten thousand pixels of it. Whether a section is open is
// the reader's, for the visit: nothing is kept.
function sectionHtml(key, title, body, openOnPhone) {
  if (!body) return '';
  const narrow = typeof window !== 'undefined' && window.matchMedia
    && window.matchMedia('(max-width: 760px)').matches;
  const open = openOnPhone || !narrow;
  return `<details class="sect" data-sect="${key}"${open ? ' open' : ''}>
    <summary><h3>${title}</h3></summary>${body}</details>`;
}

// ----------------------------------------------------------- the scores it replaced
//
// Q649 (`docs/archive/plan-confidence.md` §3.2 step 5). Each measurement a newer one replaced, as
// the server's `history` sends it: a past value labelled past, with its date, what it was
// measured with and why it was replaced, folded under the score and never in its place.
// The local application and an admin are sent every line; a visitor is sent none here and
// reads each earlier score as a *was* line under the trust line's *more* (`trustHtml`;
// §5 Q1, answered A on 2026-10-03).

function wasHtml(lines) {
  if (!Array.isArray(lines) || !lines.length) return '';
  const said = lines.map(was => {
    const number = was.index === null || was.index === undefined
      ? 'no score' : Number(was.index).toFixed(4);
    const runs = was.runs ? ` (${countWord(was.runs)} run${was.runs === 1 ? '' : 's'}${
      was.weights_version ? ', ' + was.weights_version : ''})` : '';
    const when = was.measured ? `, measured ${was.measured}` : '';
    const how = was.players ? ` with ${was.players}` : '';
    const replaced = was.replaced ? `; replaced ${was.replaced}` : '';
    const why = (was.reasons || []).length ? `: ${was.reasons.join('; ')}` : '';
    const held = was.convergence ? ` Convergence: ${was.convergence}.` : '';
    return `<li>Was ${esc(number)}${esc(runs)}${esc(when)}${esc(how)}${esc(replaced)}${
      esc(why)}.${esc(held)}</li>`;
  }).join('');
  return `<details class="was u-mt-10">
    <summary class="muted small">earlier scores (${lines.length})</summary>
    <ul class="rules muted small u-mt-6">${said}</ul>
  </details>`;
}

// ----------------------------------------------------------- a game left running here
//
// B2 of `docs/archive/plan-ux.md`. The rated games you walked away from on this game, each
// with the way back into it. Closing the tab used to lose the board: the row stayed open
// in the store and no page named it, so the game was reachable only by a URL nobody had
// kept.
//
// It sits above the play panel because starting another rated game on this game resigns
// this one. A person who cannot see the open game is one press from losing it.
//
// The local application sends no standing and keeps no rows, so it draws nothing, and so
// does a visitor -- a rated game needs an account, so there is none to have left.

// What one entry says: who is opposite, which side you are on, how far it got and when
// you last touched it. Built as one string so the sentence stays a sentence.
function carryOnLine(match) {
  const when = match.moved
    ? 'last moved ' + String(match.moved).slice(0, 10)
    : 'not moved yet';
  return `against ${shortName(match.opponent)}, playing `
    + `${match.seat === 0 ? 'first' : 'second'}, `
    + `${match.plies} turn${match.plies === 1 ? '' : 's'} played, ${when}`;
}

function carryOnHtml(game) {
  const open = (game.standing && game.standing.yourOpen) || [];
  if (!open.length) return '';
  const one = open.length === 1;
  return `<div class="panel u-mb-14">
    <h3 class="u-mt-0">Carry on</h3>
    <div class="muted small">You left ${one ? 'a rated game' : open.length + ' rated games'}
      running here. Starting another rated game on this game resigns
      ${one ? 'it' : 'them'}, and so do thirty idle days.</div>
    ${open.map(match => `<div class="row u-mt-8">
      <button class="primary" data-carry="${esc(match.id)}">carry on</button>
      <span class="muted small">${esc(carryOnLine(match))}</span>
    </div>`).join('')}
  </div>`;
}

function wireCarryOn() {
  view.querySelectorAll('[data-carry]').forEach(button => {
    button.onclick = () => {
      pending('Loading the game…');
      go('#/play/' + button.dataset.carry);
    };
  });
}

// --------------------------------------------------- play a person
//
// Q571, `docs/archive/plan-return.md` §1.4 (s1 and s2 of the mock). Beside *Play*, never in place
// of it: the members who ticked *open to a game of this*, a box to challenge any member by
// handle (suggesting members' handles as you type), a one-time challenge link, and your
// own box, off unless you tick it. All of it is `game.standing.people`, sent to a member
// alone, so a visitor and the local application draw nothing here.
//
// A challenge is one sheet: who, the seat, whether it is rated (the game's, never a
// choice), and the rules of a game between two people before the first move (test 7). A
// challenge and a link wait until they are answered or withdrawn; nothing here counts
// down (test 2). *Start a game* stays the page's one primary button: every button here
// is secondary.

function playPersonHtml(game) {
  const people = game.standing && game.standing.people;
  if (!people) return '';
  const name = esc(game.name);
  const head = `<div class="panel u-mt-14">
    <h3 class="u-mt-0">Play a person</h3>`;
  if (people.paused) {
    return `${head}<div class="small">${esc(people.pausedSays)}</div>
      ${people.mine ? openToHtml(game, people) : ''}</div>`;
  }
  const listed = people.open.length
    ? people.open.map(person => `<div class="row u-mt-6">
        <b><a href="/players/${esc(person.handle)}">${markHtml(person.mark, 20)}${
          esc(person.handle)}</a></b><span class="muted small">${esc(person.said)}</span>
        <button data-challenge="${esc(person.handle)}">challenge</button></div>`).join('')
    : `<div class="muted small">Nobody has said they're open to a game of ${name} yet.
        Challenge any member by handle, or make a link and send it to them however you
        like.</div>`;
  return `${head}
    <div class="muted small">By correspondence: you move whenever you like, there is no
      clock, and the game waits for you. ${countsWords(people.rated, game.name)}</div>
    ${makerHtml(people.maker)}
    <h4 class="u-m-10-0-4">Open to a game of ${name}</h4>
    ${listed}
    <h4 class="u-m-12-0-4">Or anyone here</h4>
    <div class="row">
      <input id="challengeWho" list="challengeMembers" placeholder="a member's handle"
        autocomplete="off" autocapitalize="off" spellcheck="false">
      <datalist id="challengeMembers">${people.members.map(handle =>
        `<option value="${esc(handle)}"></option>`).join('')}</datalist>
      <button id="challengeIt">challenge</button>
    </div>
    <div class="row u-mt-6">
      <button id="challengeLink">make a challenge link</button>
      <span class="muted small">for a member who isn't in either list; it works once</span>
    </div>
    <div id="challengeSheet"></div>
    <div id="inviteFriend">${inviteFriendHtml(people.invites)}</div>
    ${openToHtml(game, people)}
  </div>`;
}

// Q731: a friend who isn't a member yet. The friend you would invite is who you want to
// play, so the invitations `/admin` gave this person (`site/inviting.py`) are offered here
// as well as on their own page, through the same `/api/invite`. The form only for somebody
// with one left; anybody else reads the server's plain sentence saying why not and whom to
// ask. The words are the server's; nothing here counts.
function inviteFriendHtml(box) {
  if (!box) return '';
  const head = `<h4 class="u-m-12-0-4">A friend who isn't a member yet</h4>`;
  if (!(box.left > 0)) return `${head}<div class="small" id="inviteSaid">${esc(box.said)}</div>`;
  return `${head}
    <div class="small" id="inviteSaid">${esc(box.said)} Invite your friend by email, and
      once they have joined, challenge them here by their name.</div>
    <form class="row u-mt-6 u-ai-flex-end" id="inviteForm">
      <label class="small u-flex-1-1-16rem">their email address<br>
        <input id="inviteEmail" type="email" required autocomplete="off"
          autocapitalize="off" spellcheck="false" class="u-w-100p"></label>
      <button id="inviteSend" type="submit">send the invitation</button>
    </form>
    ${box.under ? `<div class="muted small u-mt-6">${esc(box.under)}</div>` : ''}`;
}

function wireInviteFriend(game) {
  const form = document.getElementById('inviteForm');
  if (!form) return;
  form.onsubmit = (event) => {
    event.preventDefault();
    const send = document.getElementById('inviteSend');
    send.disabled = true;
    api('/api/invite', { email: document.getElementById('inviteEmail').value }).then(answer => {
      game.standing.people.invites = answer.box;
      const where = document.getElementById('inviteFriend');
      where.innerHTML = inviteFriendHtml(answer.box)
        + `<div class="ok u-mt-8">${esc(answer.said)}</div>`;
      wireInviteFriend(game);
    }).catch(error => { send.disabled = false; fail(error, send); });
  };
}

// R10 (Q606): the game's maker, first, when they switched on *taking challengers* on
// their own page. Challenged through the same sheet as anybody listed; what a rated win
// earns is the server's sentence, never counted. The maker reads that they are listed.
function makerHtml(maker) {
  if (!maker) return '';
  if (maker.you) {
    return `<div class="small u-mt-10" id="makerLine">${esc(maker.said)}
      <a href="/profile">Switch it off on your page.</a></div>`;
  }
  return `<div id="makerLine" class="u-mt-10">
    <div class="row">
      <b><a href="/players/${esc(maker.handle)}">${markHtml(maker.mark, 20)}${
        esc(maker.handle)}</a></b><span class="small">${esc(maker.said)}</span>
      <button data-challenge="${esc(maker.handle)}">challenge</button></div>
    ${maker.earns ? `<div class="muted small">${esc(maker.earns)}</div>` : ''}</div>`;
}

function countsWords(rated, name) {
  return rated
    ? `It is rated on ${esc(name)}, like a rated game against the computer: the result
       moves both of your ratings here.`
    : `It is a practice game here, kept and rated nowhere, because ${esc(name)} has no
       rated computer players yet.`;
}

function openToHtml(game, people) {
  return `<label class="small u-d-block u-mt-12">
    <input type="checkbox" id="openTo"${people.mine ? ' checked' : ''}>
    List me as open to a game of ${esc(game.name)}.
    <span class="muted">Off unless you tick it. Members see your name here; nothing is
      sent to anyone.</span></label>`;
}

// The sheet, for a named member (`who`) or for a link (`who` null). The seat is named by
// the game's own sides, never by a chess word.
function challengeSheetHtml(game, who) {
  const people = game.standing.people;
  const sides = game.sides || [];
  const side = (index, fallback) => sides[index] ? esc(sides[index]) + ', ' : fallback;
  return `<div class="panel u-mt-10">
    <b>${who ? `Challenge ${esc(who)} to ${esc(game.name)}`
      : `A challenge link to ${esc(game.name)}`}</b>
    <div class="row u-mt-8">
      <label class="small">You play
        <select id="challengeSeat">
          <option value="first">${side(0, '')}moving first</option>
          <option value="second">${side(1, '')}moving second</option>
          <option value="either">either, drawn when it starts</option>
        </select></label>
    </div>
    <div class="small u-mt-6">${countsWords(people.rated, game.name)}</div>
    <ul class="rules muted small u-mt-6">${(people.rules || [])
      .map(rule => `<li>${esc(rule)}</li>`).join('')}</ul>
    <div class="row">
      <button id="challengeSend">${who ? 'send the challenge' : 'make the link'}</button>
      <button id="challengeCancel">cancel</button>
    </div>
  </div>`;
}

// What the page says once a challenge is sent or a link is made: where it waits, and for
// a link its address to send however the person likes. The site sends nothing.
function challengeSentHtml(sent, origin) {
  if (sent.link) {
    return `<div class="panel u-mt-10">
      <b>Your challenge link</b>
      <div class="row u-mt-6">
        <input id="challengeAddress" readonly value="${esc(origin + sent.link)}">
        <button id="challengeCopy">copy</button></div>
      <div class="muted small u-mt-6">It works once, for one member, and
        it waits on your page under <i>Challenges you sent</i> until it is used or you
        withdraw it. Send it however you like; the site sends nothing.</div>
    </div>`;
  }
  return `<div class="ok u-mt-10"><b>Challenge sent.</b> It waits on your
    page, under <i>Challenges you sent</i>, until ${esc(sent.to)} accepts or declines it,
    for as long as that takes. <a href="/profile">your page</a></div>`;
}

function wirePlayPerson(game, hash) {
  const panel = document.getElementById('people');
  if (!panel || !(game.standing && game.standing.people)) return;
  const sheet = document.getElementById('challengeSheet');
  const open = (who) => {
    if (!sheet) return;
    sheet.innerHTML = challengeSheetHtml(game, who);
    document.getElementById('challengeCancel').onclick = () => { sheet.innerHTML = ''; };
    const send = document.getElementById('challengeSend');
    send.onclick = () => {
      send.disabled = true;
      const body = { hash, seat: document.getElementById('challengeSeat').value };
      if (who) body.handle = who; else body.link = true;
      api('/api/challenge', body).then(sent => {
        sheet.innerHTML = challengeSentHtml(sent, location.origin);
        const copy = document.getElementById('challengeCopy');
        if (copy) copy.onclick = () => {
          const address = document.getElementById('challengeAddress');
          address.select();
          if (navigator.clipboard) navigator.clipboard.writeText(address.value).catch(() => {});
        };
      }).catch(error => { send.disabled = false; fail(error, send); });
    };
  };
  panel.querySelectorAll('[data-challenge]').forEach(button => {
    button.onclick = () => open(button.dataset.challenge);
  });
  const typed = document.getElementById('challengeIt');
  if (typed) typed.onclick = () => {
    const who = document.getElementById('challengeWho').value.trim();
    if (!who) return fail(new Error("Write a member's handle first."), typed);
    open(who);
  };
  const link = document.getElementById('challengeLink');
  if (link) link.onclick = () => open(null);
  wireInviteFriend(game);
  const tick = document.getElementById('openTo');
  if (tick) tick.onchange = () => {
    api('/api/game/' + hash + '/open', { on: tick.checked }).then(answer => {
      game.standing.people = answer.people;
      panel.innerHTML = playPersonHtml(game);
      wirePlayPerson(game, hash);
    }).catch(error => { tick.checked = !tick.checked; fail(error, tick.parentElement); });
  };
}

// --------------------------------------------------------- the games like this one
//
// M5 of `docs/archive/plan-map.md`: one meaning of near. Which games and in what order are the
// server's (`library.neighbours`), read off the map: the map's own games, most alike first
// by the map's likeness, so this list and the map's panel for the same game open on the
// same games in the same order, and the first `open` of them are the panel's *games like
// this one*. A game the map does not draw -- a draft -- is ranked against the map's games
// the same way. Nothing here ranks, re-sorts, cuts at a count of rules or works out a
// number: there is no far case, because every game has its most alike.
//
// Each row says how alike and what both have (the map's words), then what differs --
// `spec/relate.py`'s count of rules and its sentences, which explain a difference and
// order nothing -- and what the two scores make of it.

function linksHtml(game) {
  const links = game.links || [];
  if (!links.length) return '';
  const open = game.open || links.length;
  const shown = links.slice(0, open);
  const rest = links.slice(open);
  const where = !game.onMap
    ? `This game is not on the map, which draws the library's games, so these are the
       library's games most like it.`
    : game.community
      ? `On the map it is in the community <b>${esc(game.community.name)}</b>.`
      : `On the map it is unusual: in no community, because nothing in the library is
         close enough to it.`;
  // Said once rather than under every row. Where nothing in the library can be read --
  // a fresh checkout, or a corpus waiting on a re-score -- the per-row sentence was the
  // same apology twelve times over.
  const scored = links.some(link => link.gap);
  const more = rest.length
    ? `<details class="u-mt-10">
        <summary class="muted small">${rest.length} more, less alike</summary>
        ${rest.map(link => linkRow(link, scored)).join('')}
      </details>`
    : '';
  const nothing = scored ? '' : `<div class="muted small u-mt-10">No
    scores to compare yet. A link says what the difference did to the score once both
    games ${ACCOUNTS ? 'have a current score' : 'carry a measurement this checkout can read'}.</div>`;
  return sectionHtml('near', 'Games like this one', `<div class="panel">
      <div class="muted small u-mb-4">The games most like this one,
        most alike first: two games are as alike as the rare things they share${
        info('alike')}. Each says what differs between the two, by kind${
        info('differs')}. ${where}</div>
      ${shown.map(link => linkRow(link, scored)).join('')}
      ${more}
      ${nothing}
    </div>`);
}

// One game like this one: its name as the way in, how alike the two are and what both
// have, then what differs and what the two scores say. The kind of rule is a tag because
// it is the thing the map colours a difference by.
function linkRow(link, scored) {
  const apart = link.units
    ? `${link.units} rule${link.units === 1 ? '' : 's'} apart`
    : 'the same rules, written two ways';
  const kinds = (link.kinds || []).map(kind =>
    `<span class="tag">${esc(kind)}</span>`).join(' ');
  return `<div class="u-mt-10 u-pt-9 u-bt-1-solid-line">
    <div class="row u-gap-8 u-ai-baseline">
      <button class="crumb u-c-ink u-fw-600" data-link="${esc(link.hash)}">${esc(link.name)}</button>
      <span class="muted small">${esc(fmt(link.likeness))} alike${
        link.line ? ', a line on the map' : ''}</span>
    </div>
    ${(link.shares || []).length ? `<div class="small muted">both have ${
      esc(link.shares.join('; '))}</div>` : ''}
    <div class="small"><span class="muted">${esc(apart)}${
      link.units ? ':' : ''}</span> ${esc(linkWords(link))} ${kinds}</div>
    ${differsLine(link.similarity)}
    ${scored ? gapLine(link.gap) : ''}
    ${exhibitHtml(link.exhibit)}
  </div>`;
}

// What differs between the two, by kind, in the server's words: *no rules, 6 mechanisms, 1
// piece* (`spec/relate.py`'s similarity counts, said by `library.likeness_words`). Printed
// as it came: the counts and their weighing are the server's, and they order nothing.
//
// The site reads it in a visitor's words where the server sent them (G40): *what differs: 1
// rule, 28 starting squares*.
function differsLine(similarity) {
  if (!similarity || !similarity.words) return '';
  const site = ACCOUNTS && similarity.site;
  return `<div class="muted small">${site ? 'what differs' : 'by kind'}: ${
    esc(site || similarity.words)}</div>`;
}

// What differs, in words. A pair a few rules apart says all of it; one many rules apart
// says the first few and how many are left: shogi is twenty rules from the game most like
// it, and twenty clauses in one sentence is a paragraph nobody reads to find out that the
// answer is *a lot*.
function linkWords(link) {
  const parts = (link.differences || []).map(difference => difference.sentence);
  if (parts.length <= 4) return parts.join('; ');
  const rest = parts.length - 3;
  return parts.slice(0, 3).join('; ') + `, and ${rest} more`;
}

// What one rule did to the score, where both games can be read at all. Two thirds of
// one-rule pairs move the index by less than the measurement's own noise, so the panel
// says *no clear change* rather than printing a difference that is not one. A game whose
// score is stale or missing shows no number here, which is the same rule the score panel
// follows -- and it says which of the two it is, because "measured and alike" and "not
// measured" are different news. Drawn only where some link on the page has a gap;
// otherwise `linksHtml` says it once for the whole panel.
//
// Both numbers carry the runs they rest on, in the pill's words, because a library
// game's mean of four runs and a candidate's one run can sit at the two ends of one
// line. A gap whose ends do not both say how many is drawn as no gap at all.
function gapLine(gap) {
  const mine = gap ? runsShort(gap.seeds) : '';
  const theirs = gap ? runsShort(gap.otherSeeds) : '';
  if (!gap || !mine || !theirs) {
    return `<div class="muted small">No score to compare &mdash; this one has not been
      measured, or its measurement is out of date.</div>`;
  }
  const size = Math.abs(gap.gap).toFixed(3);
  const said = gap.clear
    ? `scores <b>${gap.gap > 0 ? size + ' higher' : size + ' lower'}</b>`
    : `scores about the same &mdash; no clear change`;
  return `<div class="muted small">${gap.otherIndex.toFixed(2)} (${esc(theirs)}) against
    ${gap.index.toFixed(2)} (${esc(mine)}), so it ${said}.</div>`;
}

function wireLinks() {
  view.querySelectorAll('[data-link]').forEach(button => {
    button.onclick = () => {
      pending('Loading the game…');
      go('#/game/' + button.dataset.link);
    };
  });
}

// ------------------------------------------- the game between a draft and the library
//
// Q429. The map offers *make the game in between* on a gap, and the map draws no drafts,
// so the game between a person's own draft and the library could be built and could not
// be reached. A draft's own page offers it, against the library game most like the draft
// -- the first of its own *games like this one* it may build from, by the map's likeness
// (M5 of `docs/archive/plan-map.md`) -- and says which game that is and how alike.
//
// All of it is the server's (`App._nearest`): which game, whether a game in between can
// be built and the phrase for the rule that moves, or the sentence saying why there is
// none. It arrives only where the press would be answered -- both ends are games this
// person may find -- so a stranger who opened somebody's unlisted draft by its link is
// sent none and this draws nothing. A library game is sent none either: its gaps are on
// the map. The press asks the route the map's button asks and hands what comes back to
// the editor through the map's own `mapOpenDraft`, so the two buttons open one editor
// the one way.

function betweenHtml(game) {
  const near = game.between;
  if (!near) return '';
  const count = near.units === 1 ? 'one rule' : `${near.units} rules`;
  const which = `The library game most like this one is <b>${esc(near.name)}</b>, ${
    esc(fmt(near.likeness))} alike${info('alike')} and ${count} apart: the first of the
    games like this one.`;
  if (!near.words) {
    return sectionHtml('between', 'The game in between', `<div class="panel">
        <div class="muted small">${which}</div>
        <div class="small u-mt-8">${esc(near.why)}</div>
      </div>`);
  }
  const named = digest => (digest === game.hash ? game.name : near.name);
  return sectionHtml('between', 'The game in between', `<div class="panel">
      <div class="muted small">${which}</div>
      <div class="small u-mt-8">The game in between is
        <b>${esc(named(near.from))}</b> with ${esc(near.words)} taken from
        <b>${esc(named(near.taken))}</b>.</div>
      <div class="row u-mt-8">
        <button id="between" title="open the game in between these two in the editor"
          >make the game in between</button>
      </div>
    </div>`);
}

function wireBetween(game) {
  const button = document.getElementById('between');
  if (!button) return;
  button.onclick = () => {
    const near = game.between;
    button.disabled = true;
    pending('Building the game in between…');
    api('/api/design/' + near.from + '?between=' + encodeURIComponent(near.taken))
      .then(answer => {
        clearPending();
        if (!answer.between || !answer.between.spec) {
          button.disabled = false;
          return fail(new Error(answer.between && answer.between.why
            ? answer.between.why : 'There is no game in between these two after all.'),
            button);
        }
        mapOpenDraft(answer, answer.between);
      })
      .catch(error => { button.disabled = false; fail(error, button); });
  };
}

// ------------------------------------------------------- a game that left the corpus

// What fills the play panel on a game whose rules have left the corpus. A retired game
// keeps its page and loses its board: the rules, the pieces, the score, the votes and
// the games people played are all read exactly as before, and only starting a new one is
// refused. The sentence says why, because "you cannot play this" on its own reads as a
// fault in the page.
function retiredBody(game) {
  const standing = game.standing || {};
  return `<h3 class="u-mt-0">Retired</h3>
    <div class="small">This game has been retired from the library; a newer version
      replaced it, so no new game can be started here. Everything else on this page still
      stands: the rules below are the rules that were played, and the games people played
      on it, their results and their ratings are unchanged.</div>
    ${standing.canDelete ? deleteHtml() : ''}`;
}

// The creator's button, and an admin's. Offered only where the server says so
// (`standing.canDelete`), and the server decides again when it is pressed -- a button
// the page draws is never the permission.
function deleteHtml() {
  return `<div class="u-mt-14 u-bt-1-solid-line u-pt-12">
    <div class="row">
      <button id="deleteGame" title="remove these rules for good">delete this game</button>
    </div>
    <div class="muted small u-mt-6">The rules are removed for good and
      cannot be brought back. The games played on it, their results, the ratings and the
      votes all stay, and name it a deleted game.</div>
  </div>`;
}

function wireDeleting(game, hash) {
  const button = document.getElementById('deleteGame');
  if (!button) return;
  button.onclick = () => {
    // Once. There is no undo route to offer afterwards, so the confirmation is the
    // whole of the second thought.
    const sure = window.confirm('Delete ' + game.name + ' for good?\n\nIts rules are '
      + 'removed and cannot be brought back. The games played on it stay, and name it a '
      + 'deleted game.');
    if (!sure) return;
    button.disabled = true;
    api('/api/game/' + hash + '/delete', {}).then(() => {
      // The cached detail is rules that no longer exist, and the page it drew is gone.
      delete DETAIL[hash];
      pending('Loading the library…');
      go('#/');
    }).catch(error => { button.disabled = false; fail(error, 'deleteGame'); });
  };
}

// ------------------------------------------------------------------ report a game
//
// Q503d, `docs/archive/plan-game-pipeline.md` §3.2: a quiet link under the rules. It opens to the
// three reasons the server sends (`standing.reportReasons`) and an optional note, and the
// press files one row the maintainer reads on `/admin` beside the kill switch. One report
// a person a game: once filed, the link becomes the line saying so, and the server
// refuses a second press whatever the page draws.
//
// Absent on the local application, which sends no standing, and for a visitor, who has
// nobody to report as.
function reportHtml(game) {
  const standing = game.standing;
  if (!standing || standing.signedIn !== true) return '';
  return `<div id="reporting" class="muted small u-mt-6">${
    reportBody(standing)}</div>`;
}

function reportBody(standing) {
  if (standing.yourReport) {
    return `You reported this game as ${esc(standing.yourReport.label)}. The person
      who runs the site reads every report.`;
  }
  const reasons = standing.reportReasons || [];
  return `<details>
      <summary>Report this game (spam, offensive, a copy)</summary>
      <div class="row u-mt-6 u-gap-8">
        <label>why <select id="reportReason">${reasons.map(one =>
          `<option value="${esc(one.reason)}">${esc(one.label)}</option>`).join('')}
        </select></label>
        <input id="reportNote" type="text" maxlength="500" autocomplete="off"
          placeholder="a note, if you like (which game it copies, say)"
          aria-label="a note for the person who runs the site, optional" class="u-flex-1 u-minw-12em">
        <button id="reportSend">send the report</button>
      </div>
      <div class="u-mt-4">Only the person who runs the site reads it. It takes
        nothing off the site by itself.</div>
    </details>`;
}

function wireReporting(game) {
  const button = document.getElementById('reportSend');
  if (!button) return;
  button.onclick = () => {
    button.disabled = true;
    api('/api/game/' + game.hash + '/report', {
      reason: document.getElementById('reportReason').value,
      note: document.getElementById('reportNote').value,
    }).then(answer => {
      game.standing.yourReport = { reason: answer.reason, label: answer.label };
      const panel = document.getElementById('reporting');
      if (panel) panel.innerHTML = reportBody(game.standing);
    }).catch(error => { button.disabled = false; fail(error, button); });
  };
}

// A rule of thumb is a sentence, and the sentence is the point of it -- so show the one
// you have selected rather than making the name carry the whole idea.
function wirePlans() {
  [0, 1].forEach(player => {
    const select = document.getElementById('seat' + player);
    const box = document.getElementById('plan' + player);
    if (!select || !box) return;
    const draw = () => { box.textContent = planFor(select.value); };
    select.addEventListener('change', draw);
    draw();
  });
}

// Taking this one game's measurement again. It sits under the score rather than in the
// panel about keeping, because it is the same act whatever stage the game is at -- and
// a committed ruleset had no way to be re-measured from here at all: the keeping panel
// said "nothing further to do here", which was true of promotion and false of this.
function measuringHtml(game) {
  const k = game.keeping || {};
  const stale = game.staleness && !game.staleness.current ? game.staleness : null;
  const roster = (k.roster || []).length;
  const per = k.gamesPerPairing || 4;
  const ladder = `${roster} standard opponents playing ${per} games against each other in
    every pairing${info('roster')}, which is where every score on this site comes from`;

  // One entry per repair `Staleness.needs` can ask for besides a replay. The button used
  // to offer two -- so a game whose analysis pass had changed, and whose games are
  // perfectly good, said "measure it again" and read as a replay.
  const repairs = {
    rescore: ['score it again',
      'the games still stand; only the scoring runs again, which is seconds'],
    reanalyse: ['analyse and score it again',
      'the games still stand; the pass that reads them has changed, so it runs again before the scoring does -- longer than scoring alone, shorter than playing again'],
  };

  // One sentence per state. Why a score went stale is the maintainer's business, so the
  // reason sits behind the (i) on the local application; the site's (i) says what out of
  // date means and no more (G20). On the site nobody can measure, so it says so instead.
  const local = offered('POST', '/api/test');
  let said, label, title;
  if (stale) {
    // EV2: the number below, where there is one, is the last estimate, marked
    // provisional (`provisionalMarkHtml`); this line says why and what happens next.
    const shown = isProvisional(game.score)
      && game.score.total !== null && game.score.total !== undefined;
    said = `This game's score is out of date${info('stale', ACCOUNTS ? '' : stale.summary)}${local
      ? (shown ? ', so the number below is provisional until it is measured again.'
        : ', and it has no number to show until it is measured again.')
      : (shown ? '. The number below is provisional until the library is next measured.'
        : '. It comes back when the library is next measured.')}`;
    [label, title] = repairs[stale.needs] || ['measure it again',
      'the games were played by code that has since changed, so they are played again'];
  } else if (!game.score && game.provisional) {
    // The number below is real and checked, and it is not the server's own: the sentence
    // says which, so "nobody has measured this" never sits above a number (Q312).
    said = `The site has not measured this one itself yet. The score below was measured on
      ${game.provisional.yours ? 'your' : "its designer's"} computer.`;
    label = 'measure it';
    title = 'play it against the standard opponents and score the result';
  } else if (!game.score) {
    // On the site, no score and no reason the reader can act on: the site measures
    // nothing itself yet, and saying so in its own terms is the maintainer's voice (G24).
    said = local
      ? `Nobody has measured this one yet &mdash; ${ladder}.`
      : 'No score yet.';
    label = 'measure it';
    title = 'play it against the standard opponents and score the result';
  } else {
    said = 'The score below is up to date.';
    label = 'measure it again';
    title = 'play it all again from scratch, though nothing has invalidated the score';
  }
  const state = `<div class="small">${said}</div>`;

  if (!local) return state;

  // Held for the same reason the library's sweep is, and said in the same words: a core
  // built from older sources plays by the previous build's rules without failing.
  const held = game.build ? ` disabled title="${esc(game.build)}"` : '';
  return `${state}
    <div class="row u-m-8-0-10">
      <button id="measureIt" title="${esc(title)}"${held}
        class="${stale || !game.score ? 'primary' : ''}">${esc(label)}</button>
      <span class="muted small">${game.build
        ? 'Measuring is held, because the engine changed since it was last compiled. Compile it again and restart.'
        : 'Runs in the background; you can leave the page.'}</span>
    </div>`;
}

// A number measured on somebody's own machine and checked by the server (Q312), in the
// same panel as the score and in §5's two labels (`provisionalWords`). Drawn only where
// there is no current official score, which always wins. It carries no interval, no
// components and no ladder: those are the official measurement's, and a provisional
// result seeds no rating table.
function provisionalHtml(game) {
  const p = game.provisional;
  const official = game.score && !(game.staleness && !game.staleness.current);
  if (!p || official) return '';
  const words = provisionalWords(p);
  const scored = p.index !== null && p.index !== undefined;
  const runs = runsPhrase(p.seeds);
  return `<div class="u-mt-10">
      <div class="u-fs-26 u-fw-600 u-lh-1_1">${
        scored ? fmt(p.index) : '<span class="muted">no score</span>'}</div>
      <div class="muted small">${esc(words.label)}${info('provisional')}${
        p.viable === false ? ', and ruled out' : ''}${runs ? ' &middot; ' + esc(runs) : ''}${
        p.weights ? ' &middot; ' + esc(rulesVersion(p.weights)) + info('weights') : ''}</div>
      <div class="small u-mt-6">${esc(words.why)}</div>
    </div>`;
}

function wireMeasuring(hash) {
  const button = document.getElementById('measureIt');
  if (!button) return;
  button.onclick = () => {
    button.disabled = true;
    // `force` only where there is nothing to bring up to date: pressing "measure it
    // again" on a current score has to mean something, and the server's default is to
    // do only the work the stored basis says is missing.
    const shown = DETAIL[hash] || {};
    const force = !!(shown.score && (!shown.staleness || shown.staleness.current));
    api('/api/test', { hash, force }).then(() => pollTest(hash)).catch(error => {
      button.disabled = false; fail(error);
    });
  };
  // A measurement already running -- started here, in another tab, or by a sweep.
  api('/api/test/' + hash).then(state => {
    if (state.state === 'running') pollTest(hash);
  }).catch(() => {});
}

// ------------------------------------------------------------- where it stands
//
// Q503b: the panel that was *Keeping it*. Its first fact is the game's level, the one
// word the card and the designer use too (`site/stages.py`), with the designer's four
// steps and the level's sentence, all as the server sent them. Under it, where this copy
// can keep a game (a checkout), the three actions that move a file along.
function whereHtml(game) {
  const keeps = offered('POST', '/api/promote');
  if (!game.level && !keeps) return '';
  return `<div class="panel u-mt-14" id="where" data-level="${
      esc(game.level || '')}">
      ${game.level ? `<h3 class="u-mt-0">Where it stands: ${esc(game.level)}${
        game.levelGloss ? info(game.levelGloss) : ''}</h3>
        ${trackHtml(game.track)}
        ${game.levelSays ? `<div class="small">${esc(game.levelSays)}</div>` : ''}` : ''}
      ${game.source === 'designed'
        ? `<div class="row u-mt-8">${copyLinkHtml(game.hash)}</div>` : ''}
      ${designerMarksHtml(game.designer)}
      ${shortlistHtml(game.shortlist)}
      ${roundHtml(game.round)}
      ${keeps ? `<div class="u-mt-12" id="keeping">${keepingHtml(game)}</div>` : ''}
    </div>`;
}

// D2 to D4 of `docs/archive/plan-meta-game.md`: what a designed game has reached, as the server
// counted it (`marks_from` in the site's reader). Each mark is drawn only when there is
// something to say -- a game nobody has tried reads as nothing, never as nought -- and
// each carries the sentence that says what it reads. The count names nobody (the user's
// decision B of 2026-09-14). Absent on the local application and on a game nobody
// designed here, which the server sends no `designer` for.
function designerMarksHtml(marks) {
  if (!marks) return '';
  const rules = marks.rules || {};
  const lines = [];
  const line = (said, rule, key) => lines.push(`<li data-mark="${key}" class="u-mb-2">${said}${rule
        ? ` <span class="muted">&mdash; ${esc(rule)}</span>` : ''}</li>`);
  if (marks.passesScreen) line(`<b>Passes the playability check</b>${info('playability')}`, rules.passesScreen,
      'screen');
  if (marks.tried > 0) {
    line(`<b>${marks.tried}</b> ${marks.tried === 1 ? 'person has' : 'people have'}
      tried this`, rules.tried, 'tried');
  }
  const by = marks.forkedBy || [];
  if (by.length) {
    line(`Built on by <b>${by.map(handle => personHtml(handle)).join('</b>, <b>')}</b>`, rules.forkedBy, 'forked');
  }
  if (!lines.length) return '';
  return `<ul id="designerMarks" class="small u-ls-none u-pl-0 u-m-8-0-4">${lines.join('')}</ul>`;
}

// Q503f: a trial game's standing against the shortlist's bar, every count as the server
// counted it (`site/shortlist.py`), each line met or not. Plain counts and no prompt
// (meta-game test 6): the one button is its designer's, and it works only when the
// server says the game may be nominated now; otherwise the line under it says why not.
function shortlistHtml(bar) {
  if (!bar) return '';
  const lines = (bar.checks || []).map(check => `<li class="u-mb-2"
      data-met="${check.met ? 'yes' : 'no'}"><span>${
      check.met ? '&#10003;' : '&middot;'}</span>${esc(check.says)}</li>`).join('');
  const button = bar.mine
    ? `<div class="row u-mt-6"><button id="nominateIt" class="primary"
        ${bar.nominable ? '' : 'disabled'} title="${esc(bar.nominable
          ? "put it forward for this month's members' vote" : bar.waiting)}">nominate it</button>${info('nominate')}</div>
      ${bar.waiting ? `<div class="muted small u-mt-4">${esc(bar.waiting)}</div>` : ''}`
    : '';
  return `<div id="shortlist" class="u-mt-8">
      <div class="small">${esc(bar.says)}</div>
      <ul class="small marks u-m-4-0">${lines}</ul>
      ${button}</div>`;
}

// Q503h: a nominee while members vote on it at the round (§3.2 *In a round*). The yes
// and no are the server's to offer (`site/rounds.py`); anybody else reads why not. The
// round's own page holds the rest.
function roundHtml(round) {
  if (!round) return '';
  const mine = round.yourVote;
  const control = round.mayVote
    ? `<div class="row u-mt-6">
        <button id="roundYes" class="${mine === true ? 'primary' : ''}">yes</button>
        <button id="roundNo" class="${mine === false ? 'primary' : ''}">no</button>
        ${mine === true || mine === false
          ? `<span class="muted small">You voted ${mine ? 'yes' : 'no'}.</span>` : ''}</div>`
    : `<div class="muted small">${esc(round.why)}</div>`;
  return `<div id="roundVote" class="u-mt-8">
      <div class="small">${esc(round.says)} <a href="#/round/${esc(round.round)}">The round's
        page</a>.</div>
      <div class="muted small">${esc(round.rule)}</div>
      ${control}</div>`;
}

function wireRoundVote(game) {
  [['roundYes', true], ['roundNo', false]].forEach(([id, yes]) => {
    const button = document.getElementById(id);
    if (!button) return;
    button.onclick = () => {
      button.disabled = true;
      api('/api/round/' + game.round.round + '/vote', { hash: game.hash, yes })
        .then(answer => {
          game.round = { ...game.round, yourVote: yes };
          const panel = document.getElementById('roundVote');
          if (panel) {
            panel.outerHTML = roundHtml(game.round);
            wireRoundVote(game);
            const again = document.getElementById('roundVote');
            if (again) {
              const said = document.createElement('div');
              said.className = 'small';
              said.textContent = answer.said;
              again.append(said);
            }
          }
        }).catch(error => { button.disabled = false; fail(error, button); });
    };
  });
}

function wireNominating(game) {
  const button = document.getElementById('nominateIt');
  if (!button) return;
  button.onclick = () => {
    button.disabled = true;
    api('/api/game/' + game.hash + '/nominate', {}).then(answer => {
      // The server's own words for the new level: the panel is redrawn from them.
      Object.assign(game, {
        level: answer.level, levelSays: answer.levelSays, track: answer.track,
        levelGloss: answer.levelGloss,
        shortlist: answer.shortlist,
      });
      const panel = document.getElementById('where');
      if (panel) {
        panel.outerHTML = whereHtml(game);
        const again = document.getElementById('where');
        if (again) {
          const said = document.createElement('div');
          said.className = 'small';
          said.textContent = answer.said;
          again.append(said);
        }
        wireKeeping(game.hash);
      }
    }).catch(error => { button.disabled = false; fail(error, button); });
  };
}

// ------------------------------------------------------ what becomes of a candidate

// Three actions, in the order of how much they commit you, because they are three
// different decisions and one button labelled "keep" would hide that. Testing costs
// minutes and settles nothing; archiving is a file you can delete; promotion changes
// what the whole index is calibrated against, so it is the only one that asks for a
// sentence. The refusals are the server's -- `corpus.promotion_blocker` -- and shown
// rather than enforced by a disabled button, because "why not" is the useful half.
function keepingHtml(game) {
  const k = game.keeping || {};
  const stage = k.stage || 'unknown';
  if (stage === 'committed') {
    return `<h4 class="u-m-0-0-4">Kept as part of the yardstick${info('committed')}</h4>
      <div class="muted small">It is one of the games the scoring is calibrated against,
      which is as far as a game goes here. Its score can still be taken again, above.</div>`;
  }

  const where = stage === 'archived'
    ? `It is kept in the repository${info('archived')}, and is deliberately not one of the
       games the scoring is calibrated against.`
    : `It is not kept yet${info('candidate')}. It lives in a working directory that git
       ignores, so it lasts until the next clean-up unless you keep it.`;

  const blocked = k.blocker
    ? `<div class="muted small u-mt-6">It is not ready for the yardstick
         yet (${esc(k.blocker)}).</div>`
    : '';

  return `<h4 class="u-m-0-0-4">Keeping it</h4>
    <div class="muted small">${where}</div>
    <div class="row u-mt-10">
      ${stage === 'candidate'
        ? `<button id="archiveIt" title="keep the file in the repository, without making it
            one of the games the scoring is calibrated against">keep it</button>`
        : ''}
      <button id="promoteIt" class="${k.blocker ? '' : 'primary'}"
        title="make it one of the games the scoring is calibrated against">add to the
        yardstick</button>
    </div>
    <input id="keepName" placeholder="a better name, if a machine wrote this one" class="u-mt-8">
    <textarea id="keepReason" class="u-mt-6"
      placeholder="Adding it to the yardstick? Say what this game shows that none of the others do. It is saved with the rules."></textarea>
    ${blocked}
    <div id="keepResult"></div>`;
}

function wireKeeping(hash) {
  const archive = document.getElementById('archiveIt');
  const promote = document.getElementById('promoteIt');
  const result = document.getElementById('keepResult');
  const say = html => { if (result) result.innerHTML = html; };

  const keep = (route, extra) => {
    const name = (document.getElementById('keepName') || {}).value || '';
    api(route, Object.assign({ hash, name }, extra)).then(kept => {
      // Re-read rather than patch: the game has moved along the ladder, so the panel
      // that offered the action is no longer the right panel.
      delete DETAIL[hash];
      say(`<div class="small u-mt-8">The file is now at
        <code>${esc(kept.path)}</code>.</div>`);
      setTimeout(() => go('#/game/' + hash), 600);
    }).catch(error => say(`<div class="error">${esc(errorWords(error))}</div>`));
  };

  if (archive) archive.onclick = () => keep('/api/archive', {
    note: (document.getElementById('keepReason') || {}).value || '',
  });
  if (promote) promote.onclick = () => keep('/api/promote', {
    reason: (document.getElementById('keepReason') || {}).value || '',
  });
}

// Minutes of tournament on a thread, polled exactly as a rating run is.
function pollTest(hash) {
  if (TEST_TIMER) clearInterval(TEST_TIMER);
  const tick = () => api('/api/test/' + hash).then(state => {
    const box = document.getElementById('keepState');
    if (!box) { clearInterval(TEST_TIMER); TEST_TIMER = null; return; }
    if (state.state === 'running') {
      const last = (state.log || []).slice(-1)[0] || 'starting';
      box.innerHTML = `<div class="muted small u-mt-8">It is measuring
        against the standard opponents, at ${esc(last)} after
        ${Math.round(state.seconds || 0)}s. It runs in the background, so the page stays
        usable.</div>`;
      return;
    }
    clearInterval(TEST_TIMER); TEST_TIMER = null;
    if (state.state === 'failed') {
      box.innerHTML = `<div class="error">${esc(state.error || 'the measurement failed')}</div>`;
      return;
    }
    // The blocker and the score both moved, so the panel is rebuilt from the server.
    delete DETAIL[hash];
    go('#/game/' + hash);
  }).catch(() => { clearInterval(TEST_TIMER); TEST_TIMER = null; });
  TEST_TIMER = setInterval(tick, 2000);
  tick();
}

function wireRating(hash) {
  const button = document.getElementById('rate');
  if (!button) return;
  button.onclick = () => {
    button.disabled = true;
    api('/api/rate', { hash }).then(() => pollRating(hash)).catch(error => {
      button.disabled = false;
      fail(error);
    });
  };
}

// Minutes of tournament on a thread, so the page polls exactly as it does for a design
// run. When it lands the game's detail is patched in place rather than refetched: the
// rules have not changed and the picker is holding a chosen seat.
function pollRating(hash) {
  if (RATING_TIMER) clearInterval(RATING_TIMER);
  const tick = () => api('/api/rating/' + hash).then(state => {
    const box = document.getElementById('strength');
    if (!box) { clearInterval(RATING_TIMER); RATING_TIMER = null; return; }
    if (state.state === 'running') {
      // Elapsed, not a bar: the core plays its games in one piece and reports nothing
      // until they are all back. Then the runner's callback sets `played` to them and
      // counts on as each of Fairy-Stockfish's games finishes, one at a time (Q446) --
      // see ratingWait for how long each is.
      const wait = state.played && state.played < state.total
        ? `${state.played} are done, and the other ${state.total - state.played} are
           Fairy-Stockfish's, one at a time. All told it ${ratingWait(false)}.`
        : `It ${ratingWait(true)}.`;
      box.innerHTML = `<div class="muted small">It is playing ${state.total} games,
        ${Math.round(state.seconds || 0)}s so far. ${wait} It runs in the background, so
        the page stays usable and you can start a game while it finishes.</div>`;
      return;
    }
    clearInterval(RATING_TIMER); RATING_TIMER = null;
    if (state.state === 'failed') {
      box.innerHTML = `<div class="error">${esc(state.error || 'the rating run failed')}</div>`;
      return;
    }
    if (state.table && DETAIL[hash]) {
      DETAIL[hash].pickerRatings = state.table;
      const game = DETAIL[hash];
      box.innerHTML = strengthHtml(game);
      wireRating(hash);
      // The seat pickers carry the ratings in their labels, so they are redrawn too --
      // keeping whichever opponents were chosen, because losing a chosen seat to a
      // background task finishing would be its own small betrayal.
      const chosen = [0, 1].map(i => (document.getElementById('seat' + i) || {}).value);
      const holder = document.querySelector('.sides');
      if (holder) {
        holder.outerHTML = sidePicker(game, ROSTER, ratingTable(game));
        [0, 1].forEach(i => {
          const select = document.getElementById('seat' + i);
          if (select && chosen[i]) select.value = chosen[i];
        });
        wirePlans();
      }
    }
  }).catch(() => { clearInterval(RATING_TIMER); RATING_TIMER = null; });
  RATING_TIMER = setInterval(tick, 1500);
  tick();
}

// How long rating the picker takes, as the button promises it and as the running
// message repeats it. It used to promise about a minute whatever the game, and with
// Fairy-Stockfish in the picker the run took two minutes on gardner, seven on crazyhouse
// and chess and more than ten on shogi (Q432): the core cannot drive an outside engine,
// so a schedule with one in it was played one game at a time. The core's games now go
// to the core (`tournament.runner`, Q439), and what is left one at a time is
// Fairy-Stockfish's twenty. Measured on the 16-core box at ten threads, where the button
// uses all but one core:
//   without it   dobutsu 4s, rps-v2 30s, xiangqi 98s, courier's own players 147s
//   with it      gardner 71s, crazyhouse 180s, chess 183s, courier 562s
// Courier is the long pole: 12x8, long games, twenty seconds a game against the outside
// engine. The words bound what was measured rather than guess per board, because the
// board's size is a poor predictor (rps-v2 is 9x9 and takes a third of xiangqi's time).
// `store` is the roster's own answer to where an opponent came from; Fairy-Stockfish is
// the one player in any roster the core cannot run (no `native_spec`). `why`
// leaves the reason off where the sentence before it has just given it.
function ratingWait(why) {
  return ROSTER.some(o => o.store === 'fairy')
    ? `takes a minute on a small board and up to ten on a large one${why
        ? ", because Fairy-Stockfish's games are played one at a time" : ''}`
    : 'takes under a minute on a small board and up to three on a large one';
}

// The scale the local application's table is on, named once: the tournament's, worked out
// there once. The site draws no such table: a game's page there has its leaderboard, on the
// one scale people are rated on (`docs/reference/plain-words.md` §3.3; the audit's G11 and G14).
const LADDER_SCALE = 'in the tournament';

// How strong each opponent is *in this game*, which is the thing a node budget cannot
// tell you: 6,400 nodes is +770 Elo on chess and level with random on racing kings. The
// local application's panel; the site's is `leaderboardHtml` (L2 of
// `docs/archive/plan-leaderboard.md`), which replaced this one there.
//
// Exactly one table is ever shown -- see ratingTable -- and it says which, because the
// rated ladder's table and the picker's own are two fits over two corpora and reading a
// number out of each in one column would invent a comparison nobody made.
//
// A table is a snapshot of whoever was in the picker when it was worked out, and the
// picker is live: an engine installed since, a rule of thumb proposed since, a
// valuation accepted since is offered above and has no row here. The panel used to
// draw the rows it had and nothing else, so an opponent you could start a game against
// this minute was simply missing from "how hard each opponent is" with no sign that it
// had ever been asked about -- Fairy-Stockfish on chess, whose table predates the
// engine. Now the difference between the roster and the table is named, both ways.
//
// Only for the picker's own table. The rated ladder's rows are the canonical agents --
// `minimax:nodes=6400` and its rungs -- which are not picker entries at all, so
// a diff against the roster there would call the whole ladder "no longer offered" and
// the whole picker "not rated yet"; its note says the honest thing in one sentence.
//
// One number a row: the tournament's, the table's own and fixed.
function strengthHtml(game) {
  const table = ratingTable(game);
  const picker = !!(table && table.kind === 'picker');
  const rated = new Set(picker ? table.standings.map(s => s.name) : []);
  // In the picker and not in the table: rated by nobody yet.
  const unrated = picker
    ? ROSTER.filter(o => o.agent !== 'human' && !o.practiceOnly && !rated.has(o.agent))
    : [];
  // Rating plays a tournament, from the button below where the server offers it.
  const rates = offered('POST', '/api/rate');
  const button = rates ? `<div class="row u-mt-10">
      <button id="rate">${picker
        ? (unrated.length
            ? `work them out again (${unrated.length} not rated yet)`
            : 'work them out again')
        : 'work out how hard each one is'}</button>
      <span class="muted small">plays them against each other, which ${ratingWait(true)}</span>
    </div>` : '';
  // A searched table or rule of thumb the roster left out as stale (Q142), in the server's
  // own sentence: what is stale, and the command that repairs it. It is refused rather
  // than played as the built-in objective under its own name; `opponents.refused_stores`
  // says why. Here because rating the picker is where one of them used to fail, with a
  // message that blamed the agent kind.
  const refused = refusedHtml(game);
  if (!table) {
    return `<div class="muted small">Nobody has been rated on this game yet. It has to be
      worked out per game and cannot be assumed, because the same opponent is 770 points
      above a random mover on chess and level with it on racing kings.</div>`
      + refused + button;
  }
  const runs = runCount(table.seeds);
  const note = picker
    ? `This table holds every opponent that was in the picker when it was worked out, over
       ${table.games} games. It is a quick estimate. Most of them played about twenty
       games, so read these to the nearest hundred and expect ties.`
    : `This table holds only the standard opponents${info('roster')}, from the
       ${table.games} games that scored this one${runs > 1
         ? `, across ${runs} runs -- each opponent's number here is its
            mean over them` : ''}. The ones with a plan are not in it${
       rates ? ', so press below to rate everything' : ''}.`;
  const scales = `This copy keeps no accounts and rates no people, so each opponent has one
     number, its number ${LADDER_SCALE}${info('ladderScale')}.`;
  const rows = table.standings.map(s => {
    const entry = ROSTER.find(o => o.agent === s.name);
    // The live roster's name for an opponent wins over the one stored with the table.
    // A rating table is a set of *numbers* about agent strings; the words beside them
    // are this interface's, and a table written before a label was reworded should not
    // keep printing the old one next to a game you can start right now.
    const label = (entry && entry.label) || s.label || shortName(s.name);
    const group = (entry && entry.group) || s.group || '';
    // The rungs every board shares need no group beside them; a stored table names
    // theirs as it was worded when it was written.
    const ladderRow = (entry && entry.store === 'ladder') || s.group === 'search depth'
      || s.group === 'computer players, weakest first';
    // In the picker's table and not in the picker: a rule of thumb whose set was
    // regenerated, a valuation that has since been refused. Its number still stands; it is
    // just not a game you can start, and the row says so rather than looking like one.
    const gone = picker && !entry ? ' <span class="muted small">(no longer available)</span>' : '';
    const number = `<b>${s.rating > 0 ? '+' : ''}${Math.round(s.rating)}</b>`;
    return `<tr>
        <td>${esc(label)}${gone}</td>
        <td class="muted small">${esc(ladderRow ? '' : group)}</td>
        <td class="u-ta-right">${number}</td>
        <td class="muted small u-ta-right">${
          s.games === undefined || s.games === null ? '' : s.games}</td>
      </tr>`;
  }).join('');
  const missing = unrated.length && !rates
    ? `<div class="muted small u-mt-8">${unrated.length} of the computer
        players you can pick ${unrated.length === 1 ? 'has' : 'have'} no rating on this game
        yet.</div>`
    : unrated.length
    ? `<div class="muted small u-mt-8">${
        unrated.map(o => `<b>${esc(o.label)}</b>`).join(', ')} ${
        unrated.length === 1 ? 'is' : 'are'} not rated yet, because ${
        unrated.length === 1 ? 'it was' : 'they were'} not in the picker when this table
        was worked out. Press below to rate everything.</div>`
    : '';
  // The heading is the whole label. The column is about fifty pixels wide beside a
  // wrapping opponent name, so a second line under it -- "zero is the middle", "people
  // are rated here too" -- broke into four and pushed the numbers apart; the sentence
  // above the table says both of those things in full and the (i) says the rest.
  const column = (words, gloss) =>
    `<th class="u-ta-right">${words}${info(gloss)}</th>`;
  return `<div class="muted small">${note}</div>
    <div class="muted small u-mt-6">${scales}</div>
    <table class="ratings"><thead><tr>
      <th>opponent</th><th></th>
      ${column(LADDER_SCALE, 'ladderScale')}
      <th class="u-ta-right">games</th>
    </tr></thead><tbody>${rows}</tbody></table>` + missing + refused + button;
}

// The stored players the roster left out, each in the server's own sentence (G19): under
// the local application's strength table and under the site's leaderboard alike.
function refusedHtml(game) {
  return (game.refusedStores || []).map(line =>
    `<div class="muted small u-mt-8">${withCommands(line)}</div>`).join('');
}

// A server sentence with its commands in backticks, as the page sets code: escaped
// first, so nothing in the sentence is markup but the span this adds.
function withCommands(line) {
  return esc(line).replace(/`([^`]+)`/g,
    '<span class="u-ff-mono">$1</span>');
}

// ------------------------------------------------------------------- the leaderboard
//
// L2 of `docs/archive/plan-leaderboard.md` (§2.1, §2.5, §3): on the site, the game's one rating
// table, computer players and people together, where the local application keeps its
// strength table above. Everything in a row is the server's (`site/leaderboard.py`, sent as
// `standing.leaderboard`): who is on it, the order, the rounded number, the `?`, the count,
// which rows are the strongest, no longer offered, or the reader's. The page sorts, rounds
// and decides nothing; it draws the rows in the order they came and the sentences around
// them. Null on the local application, which has no table of people, and there the
// strength table is drawn as it always was. On the site a game the server sent no table
// for reads as one whose computer players are not rated yet, never as the local panel.
function boardOf(game) {
  if (!ACCOUNTS) return null;
  return (game && game.standing && game.standing.leaderboard)
    || { rated: false, rows: [], people: 0, unrated: [], waiting: [], library: false };
}

// §2.5: before the computer players are rated the game is practice, and the panel says so
// with their names, weakest first, and no numbers -- never a table of zeros or a blank
// column. A library game's ladder is rated on our own machine. A member's game is rated on
// the site when it is published, the five standard players first (L5), and `board.box`
// says how far that has got; a game a cap refused waits for our own machine, with no time
// promised.
const UNRATED_SAYS = `This game's computer players have not been rated yet, so every game
  here is practice for now.`;
const UNRATED_WAIT = 'They are rated on our own machine; this usually takes a day.';
const BOX_SAYS = {
  draft: 'They are rated on this site when the game is published.',
  rating: `The five standard ones are being rated on this site now; this usually takes a few
    minutes.`,
  slow: `Rating them took longer than this site allows, so they are waiting to be rated on
    our own machine.`,
  large: `Rating them needed more memory than this site allows, so they are waiting to be
    rated on our own machine.`,
  waiting: `They could not be rated on this site, so they are waiting to be rated on our own
    machine.`,
};
const NOBODY_YET = `Nobody has played a rated game here yet. Your first one puts you on
  this table, starting at 1500.`;

// The players with no row yet (`board.unrated`). On a member's game rated here at publish
// they are the stronger ones, who join when our own machine has rated them (L6); anywhere
// else, a player offered since the ladder was rated.
function unratedLine(board) {
  const unrated = board.unrated || [];
  if (!unrated.length) return '';
  const names = unrated.map(name => `<b>${esc(name)}</b>`).join(', ');
  const one = unrated.length === 1;
  if (board.box === 'rated') {
    return `The five standard computer players were rated on this site when the game was
      published. ${names} ${one ? 'joins' : 'join'} this table when our own machine has
      rated ${one ? 'it' : 'them'}; until then a game against ${one ? 'it' : 'them'} is
      practice.`;
  }
  return `${names} ${one ? 'has' : 'have'} no rating on this game yet, so ${
    one ? 'it is' : 'they are'} not on it.`;
}

function leaderboardHtml(game) {
  const board = boardOf(game);
  const refused = refusedHtml(game);
  const line = (words, top) =>
    `<div class="muted small${top ? ` ${styleClass(`margin-top:${top}px`)}` : ''}">${words}</div>`;
  if (!board.rated) {
    const waiting = board.waiting || [];
    const when = board.library ? UNRATED_WAIT : (BOX_SAYS[board.box] || '');
    return line(`${UNRATED_SAYS}${when ? ' ' + when : ''}`)
      + (waiting.length ? `<table class="ratings" id="waiting"><thead><tr>
          <th>its computer players, weakest first</th></tr></thead><tbody>${
          waiting.map(name => `<tr><td>${esc(name)}</td></tr>`).join('')}</tbody></table>`
        : '')
      + refused;
  }
  const rows = board.rows || [];
  const settling = rows.some(row => row.settling);
  const who = board.people
    ? 'Everyone who has finished a rated game here, and every computer player'
    : 'Every computer player on this game';
  const said = `${who}, strongest first, on the scale people are rated on. The computer
    players started from about twenty games each against one another, and rated games
    against people move them.${settling
      ? ' A number with a ? is still settling, over about the first ten games.' : ''}`;
  const body = rows.map(row => {
    const name = row.computer ? esc(row.name) : personHtml(row.name);
    const marks = [
      row.you ? 'you' : '',
      row.top ? 'the one to beat' : '',
      row.gone ? 'no longer offered' : '',
    ].filter(Boolean);
    return `<tr${row.you ? ' class="you"' : ''}>
        <td>${name}${marks.length
          ? ` <span class="muted small">(${marks.join(', ')})</span>` : ''}</td>
        <td class="u-ta-right u-ws-nowrap"><b>${esc(row.rating)}</b>${
          row.settling ? '<span title="still settling">?</span>'
          : settling ? '<span aria-hidden="true" class="u-vis-hidden">?</span>' : ''}</td>
        <td class="muted small u-ta-right">${esc(row.games)}</td>
      </tr>`;
  }).join('');
  const unrated = unratedLine(board);
  const missing = unrated ? line(unrated, 8) : '';
  return line(said) + `<table class="ratings"><thead><tr>
      <th>player</th>
      <th class="u-ta-right u-ws-nowrap">rating${info('ratedHere')}</th>
      <th class="u-ta-right">rated games here</th>
    </tr></thead><tbody>${body}</tbody></table>`
    + (board.people ? '' : line(NOBODY_YET, 8)) + missing + refused;
}

// The only measurement on this page that is not the ladder grading itself.
//
// Two things are said before the rating and on purpose. Games begun and never finished
// come first, because a person who closes the tab when they are losing has a record
// this store cannot bound from below and no arithmetic downstream can repair that. And
// a clean sweep in either direction is reported as an arrow rather than a number: the
// results are a bound, and the only thing making the fit finite there is the prior.
function yourRecordHtml(mine) {
  const shell = inner => `<div class="panel u-mt-14" id="yourRecord">
      <h3 class="u-mt-0">How you do against them</h3>${inner}</div>`;
  if (!mine || !mine.started) {
    return shell(`<div class="muted small">You have not played this one yet. Everything
      above is the computer grading itself &mdash; the same handful of players answering
      every question about how good the game is. Your result is the only number on this
      page that comes from outside, and it is kept automatically, win or lose. Nothing
      about the game's score is affected by it.</div>`);
  }
  const anchor = mine.anchor;
  const score = mine.score === null || mine.score === undefined
    ? '' : `<b>${(mine.score * 100).toFixed(0)}%</b> scored`;
  const abandoned = mine.abandoned
    ? `<span class="muted small">${mine.abandoned} left unfinished, which the rest cannot
       account for</span>`
    : '';
  const rows = (mine.byOpponent || []).map(entry => {
    const found = ROSTER.find(o => o.agent === entry.agent);
    return `<tr>
      <td>${esc((found && found.label) || entry.agent)}</td>
      <td class="muted small u-ta-right">${
        entry.rating === null || entry.rating === undefined
          ? '' : (entry.rating > 0 ? '+' : '') + Math.round(entry.rating)}</td>
      <td class="u-ta-right"><b>${(entry.score * 100).toFixed(0)}%</b></td>
      <td class="muted small u-ta-right">${entry.games}</td>
    </tr>`;
  }).join('');
  const table = rows ? `<table class="ratings"><thead><tr>
      <th>opponent</th><th class="u-ta-right">its rating${info('elo')}</th>
      <th class="u-ta-right">you scored</th><th class="u-ta-right">games</th>
    </tr></thead><tbody>${rows}</tbody></table>` : '';

  let verdict;
  if (!anchor) {
    verdict = `<div class="muted small u-mt-8">None of the opponents you
      played has a rating on this game yet, so there is nothing to place you against.
      Work them out above and this fills in &mdash; your games are already kept.</div>`;
  } else if (!anchor.bounded) {
    const up = anchor.swept === 'up';
    verdict = `<div class="u-mt-8">You ${up ? 'won' : 'lost'} all
      ${anchor.games} games against rated opponents. That puts you
      ${up ? 'above' : 'below'} all of them and says nothing about by how much &mdash;
      these games cannot answer that. Try one ${up ? 'harder' : 'easier'}.</div>`;
  } else {
    const gap = anchor.top_above_human;
    const sits = gap >= 0
      ? `<b>${Math.round(gap)} points above you</b>`
      : `<b>${Math.round(-gap)} points below you</b>`;
    verdict = `<div class="u-mt-8">You rate
      <b>${anchor.rating > 0 ? '+' : ''}${Math.round(anchor.rating)}</b> at this game,
      over ${anchor.games} games against rated opponents. The strongest of them
      (${esc((ROSTER.find(o => o.agent === anchor.top) || {}).label || anchor.top)})
      is ${sits} &mdash; you would expect to score
      ${(anchor.expected_against_top * 100).toFixed(0)}% against it.
      <div class="muted small u-mt-6">Only your own rating is worked out
      from these games; every opponent keeps the one it already had, and none of this
      touches the game's score.</div></div>`;
  }
  return shell(`<div class="row small u-gap-8">
      <span><b>${mine.finished}</b> finished</span>
      ${score ? '<span>&middot;</span><span>' + score + '</span>' : ''}
      ${abandoned ? '<span>&middot;</span>' + abandoned : ''}
    </div>${table}${verdict}`);
}

// Which kind of game the next one will be. Absent on the local application, which keeps
// nothing and has no table to be rated on.
//
// §4's two modes. Sandbox has undo and restart and is read by no statistic; leaderboard
// is rated and refuses undo. Sandbox is the default, because a game that counts is
// something a person chooses and never something they get by leaving a field alone. An
// unrated game has no computer players on a table, so there is nothing to rate a result
// against and the choice is not offered.
//
// Walking away from a rated game leaves its row open. What ends it is the thirty-day
// idle sweep or starting another rated game on the same game, and the option says both;
// the *carry on* list above is the way back into one you left (B1 and B2 of
// `docs/archive/plan-ux.md`).
//
// A visitor who has not signed in is offered sandbox alone. A rated result moves a table
// whose rows are people, so the server refuses one without a name, and a menu offering
// what the server will refuse is worse than a sentence saying so. A visitor is also told
// that the game is not kept for them (the first-visit study's F5): two readers played
// signed out, made an account, and found a profile reading *0 games tried* that nothing
// had warned them of.
//
// The menu's options are two words each and the sentence under it says what the chosen
// one means (F4). A select's option cannot wrap, so the rated option's sentence was cut
// off at 375px (*yes — rated, no undo; an unfinishe…*); a line under the menu wraps.
//
// The choice is remembered (F4): sandbox is the default only until a person has chosen.
// A club player's second game was a sandbox game because the menu went back to *no*.
// What is kept is the last mode a game was started in, in this browser, and nothing else.
const UNKEPT = `A game played without an account isn't kept: it won't be on the profile
  you make later.`;

// The mode's value stays `sandbox`, which is what the server and the record call it; a
// person reads *practice* (`docs/reference/plain-words.md` §2; the audit's G5).
const MODE_SAYS = {
  sandbox: 'This is a practice game: undo and restart are on, and it counts for nothing.',
  leaderboard:
    'This one is rated, no undo; an unfinished game is resigned after thirty idle days or when you start another.',
};

const MODE_KEY = 'cvs-play-mode';

// The roster entry among `seats` that no game is rated against, or null (Q683). The
// server says which (`practiceOnly`); the page only reads it.
function practiceOnlyIn(seats) {
  return ROSTER.find(o => o.practiceOnly && seats.includes(o.agent)) || null;
}

// What the line under the mode menu says while such a player is chosen.
function practiceOnlySays(entry) {
  return `A game against ${entry.name} is always practice: it is not rated, so undo and
    restart are on and it counts for nothing.`.replace(/\s+/g, ' ');
}

function lastMode() {
  try {
    const kept = window.localStorage.getItem(MODE_KEY);
    return MODE_SAYS[kept] ? kept : 'sandbox';
  } catch (error) {
    return 'sandbox';
  }
}

function keepMode(mode) {
  try { window.localStorage.setItem(MODE_KEY, mode); } catch (error) { /* not kept */ }
}

function modePicker(game) {
  const standing = game.standing;
  if (!standing) return '';
  const visitor = standing.signedIn === false ? ` ${UNKEPT}` : '';
  // A preview (Q931) is practice whatever its computer players are: it has no leaderboard.
  if (game.preview) {
    return `<div class="muted small u-mt-8">A preview is played as
      practice${info('practice')}: undo and restart are on, and nothing counts towards a
      rating or Unbeaten.${visitor}</div>`;
  }
  if (!standing.rated) {
    // §2.5 of `docs/archive/plan-leaderboard.md`: the leaderboard's own sentence, said here too.
    return `<div class="muted small u-mt-8">This game's computer players
      have not been rated yet, so every game here is practice${info('practice')} for now
      &mdash; undo and restart are on, and nothing counts towards a rating or
      Unbeaten.${visitor}</div>`;
  }
  if (standing.signedIn === false) {
    return `<div class="muted small u-mt-8">This is a practice
      game${info('practice')}: undo and restart are on, and it counts for nothing.
      Sign in to play a game that counts towards the ratings and Unbeaten.${visitor}</div>`;
  }
  // The clamp stays: a select is as wide as its widest option, and without it the menu
  // once set the column's width and the page scrolled sideways at 375px. A first game
  // here is practice (K3), whatever this browser last chose.
  const chosen = firstSeat(game) ? 'sandbox' : lastMode();
  const option = (value, words) => `<option value="${value}"${
    value === chosen ? ' selected' : ''}>${words}</option>`;
  return `<label class="pick u-mt-8">does this one count
    <select id="mode" class="u-maxw-100p">
      ${option('sandbox', 'no &mdash; practice')}
      ${option('leaderboard', 'yes &mdash; rated')}
    </select></label>
    <div class="muted small u-mt-4" id="modeSays">${MODE_SAYS[chosen]}</div>`;
}

// ------------------------------------------------------------------ a first game
//
// K3 (`docs/archive/plan-understanding.md` §4.2 steps 3-4): the first game a person starts on a
// game seats the gentlest computer player on its ladder above random, which the server
// read off the game's rating table as the front page's first step does (`firstGame`,
// `web/launch.py`); it is practice, and the board opens with the *rules* tab beside it.
// *A harder one* is one tap away. On a game far from chess (the server's reasons) the
// page offers *watch two computer players first*. *First* is this browser's: a game
// started here is noted (`PLAYED_KEY`), and nothing reaches the server.
const PLAYED_KEY = 'cvs-played';

function playedHere() {
  try {
    const kept = JSON.parse(window.localStorage.getItem(PLAYED_KEY) || '[]');
    return Array.isArray(kept) ? kept : [];
  } catch (error) {
    return [];
  }
}

function notePlayed(hash) {
  try {
    const kept = playedHere();
    if (!kept.includes(hash)) {
      kept.push(hash);
      window.localStorage.setItem(PLAYED_KEY, JSON.stringify(kept.slice(-500)));
    }
  } catch (error) { /* not kept */ }
}

function gentlestOf(game) {
  const first = game && game.firstGame;
  return first && first.gentle ? first.gentle.agent : null;
}

// The gentlest player's agent where this is the browser's first game on this game, else
// null. Only a player the picker offers: a roster that lost it seats the usual one.
function firstSeat(game) {
  const gentle = gentlestOf(game);
  if (!gentle || playedHere().includes(game.hash)) return null;
  return ROSTER.some(o => o.agent === gentle) ? gentle : null;
}

function firstGameHtml(game) {
  const first = game && game.firstGame;
  if (!first || !first.gentle) return '';
  const far = first.far || [];
  const watch = far.length ? `<div class="muted small u-mt-6">This game is far
      from chess: ${esc(far.join('; '))}.</div>
    <div class="row u-mt-6"><button id="watchFirst"
      title="${esc(watchTitle())}">watch two computer players first</button></div>` : '';
  if (!firstSeat(game)) return watch ? `<div class="firstGame">${watch}</div>` : '';
  return `<div class="firstGame">
    <div class="small">Your first game here is against <b>${esc(first.gentle.name)}</b>, the
      gentlest computer player on this game, as practice, with the rules open beside the
      board.</div>
    ${first.harder ? `<div class="row u-mt-6"><button id="firstHarder"
      >a harder one: ${esc(first.harder.name)}</button></div>` : ''}
    ${watch}
  </div>`;
}

function wireFirstGame(game, hash) {
  const first = game.firstGame;
  const harder = document.getElementById('firstHarder');
  const second = document.getElementById('seat1');
  if (harder && second && first && first.harder) harder.onclick = () => {
    second.value = first.harder.agent;
    second.dispatchEvent(new Event('change'));
  };
  const watch = document.getElementById('watchFirst');
  if (watch) watch.onclick = () => {
    pending('Starting the game…');
    api('/api/session', { hash, seats: [WATCH_AGENT, WATCH_AGENT] })
      .then(open_).catch(error => fail(error, 'watchFirst'));
  };
}

// What the start button starts, in its own words (F4): *start a game* said nothing about
// whether it would count or who was opposite, and a reader's first game was sandbox
// against *medium* without its noticing. Read off the two menus and the mode as they
// stand; the labels are the picker's own (`shortName`).
function startWords(seats, mode) {
  const people = seats.filter(seat => seat === 'human').length;
  const kind = mode === 'leaderboard' ? 'rated ' : mode === 'sandbox' ? 'practice ' : '';
  if (people === 1) {
    return `start a ${kind}game against ${shortName(seats.find(seat => seat !== 'human'))}`;
  }
  if (people === 2) return `start a ${kind}game for two at this board`;
  return `start a game: ${shortName(seats[0])} against ${shortName(seats[1])}`;
}

// The tag, the target under it, and what you have done here. The target is named whether
// or not the tag is still on: "Unbeaten" with nothing named is a mystery rather than a
// motivator, and §5 wants the thing to beat visible from the start.
function standingHtml(game) {
  const standing = game.standing;
  if (!standing) return '';
  // The badges that sit on this game go in the same panel, under the lines about
  // people. `wireBadges` fills the box with text nodes after the view is in place.
  const shell = (title, inner) => `<div class="panel u-mt-14">
      <h3 class="u-mt-0">${title}${info('unbeaten')}</h3>${inner}
      <div id="gameBadges"></div></div>`;
  if (!standing.rated) {
    return shell('Unbeaten', `<div class="muted small">Its computer players have no
      ratings yet, so there is no rated play and no Unbeaten.</div>`);
  }
  const top = standing.top || [];
  const target = top.map(agent => esc(shortName(agent))).join(' and ');
  const rating = standing.topRating ? Math.round(standing.topRating) : null;
  // One scale (`docs/reference/plain-words.md` §3.3; the audit's G11): the number people are
  // rated on, the one that moves. The tournament's number for the same player stays on the local
  // application's strength table, which draws no panel like this one.
  const first = standing.firstClear;
  const head = standing.unbeaten
    ? `<div><b>Nobody has beaten ${target} here.</b> Win a rated game against
       ${top.length > 1 ? 'each of them' : 'it'} and your name stays on this game.</div>`
    : `<div><b>First beaten by ${first && first.handle ? personHtml(first.handle) : 'somebody'}</b>${
       first && first.when ? ' on ' + esc(first.when.slice(0, 10)) : ''}, playing
       ${first && first.seat === 0 ? 'first' : 'second'}.
       Its strongest computer player${info('strongest')} is ${target}.</div>`;
  const where = top.length > 1
    ? `They are tied as its strongest computer players${rating
       ? `, rated <b>${rating}</b> here${info('ratedHere')}` : ''}, so beating this game
       means beating each of them.`
    : rating
    ? `Its strongest computer player is rated <b>${rating}</b> here${info('ratedHere')}.`
    : 'Its strongest computer player has no rating here yet.';
  return shell(standing.unbeaten ? 'Unbeaten' : 'First to beat the strongest',
    `${head}${where
      ? `<div class="muted small u-mt-4">${where}</div>` : ''
    }${clearsHtml(standing)}${firstPlayedHtml(standing)}${highestHtml(standing)}${
      rungHtml(standing.yourRecord)}${ladderRowHtml(standing.yourLadder)}`);
}

// Q653 (`docs/reference/opponent-line.md` §1): whether people's play has beaten this game's
// strongest computer player, in one of three states the server reads live from people's
// rated games (`site/opponent_line.py`). Under the computer players' table, beside the
// trust line and never inside it: one is about the challenge, the other about the score.
// The page draws what it is sent; the local application keeps no people's results and
// sends no line. H7 (`docs/reference/human-data.md` §3.2): each computer player's place among
// the people rated here joins the same *more*, worded by the server and sent only once
// five people are established on the game's table (`site/people_scale.py`).
function opponentLineHtml(standing) {
  const line = standing && standing.opponentLine;
  if (!line || !line.state) return '';
  const scale = (standing.peopleScale && standing.peopleScale.said) || [];
  const more = [...(line.more || []), ...scale];
  const state = line.state.charAt(0).toUpperCase() + line.state.slice(1);
  return `<div class="opponentLine" id="opponentLine"><strong>${esc(state)}.</strong> ${
    esc(line.said)}${more.length ? `
    <details class="trustMore"><summary>more</summary>
      <ul class="rules small">${more.map(said => `<li>${esc(said)}</li>`).join('')}</ul>
    </details>` : ''}</div>`;
}

// R7 (`docs/archive/plan-return.md` §3 C3): every computer player on this game, weakest first,
// each marked with your best result against it (*won*, *drew*, *not yet*), from your
// finished rated games. The server folds the marks (`unbeaten.ladder_from`); a row of
// *won* is the whole ladder. Signed in and on a rated game only, and nothing to draw
// when the server sends none.
function ladderRowHtml(steps) {
  if (!steps || !steps.length) return '';
  return `<div class="muted small u-mt-8" id="yourLadder">Your best result
    against each computer player here, weakest first:</div>
    <ul class="ladderRow">${steps.map(step => `<li class="tag ${
      step.mark === 'won' ? 'won' : step.mark === 'drew' ? 'drew' : 'notYet'}">${
      esc(step.name)}: ${esc(step.mark)}</li>`).join('')}</ul>`;
}

// ------------------------------------------------- the meta-game's three lines
//
// Box B1 of `docs/archive/plan-meta-game.md`. Each is a record about people rather than
// about the game, and each is absent when there is nothing to say: nobody has played it,
// nobody is established on its table, nobody has beaten the top. A blank line would read
// as a nought, and `docs/archive/plan-meta-game.md` §2.1's third rule is that absent is not zero.
//
// All three come from the server -- `SiteApp._standing` -- and the page sorts nothing.
// The highest-rated person is whoever the server says, never a sort this page does over
// `siteRatings`, which holds the computer players and no people at all. The local
// application has no accounts, so it sends no standing and draws none of this.

// Everyone who has beaten the top, in the order they did it, under the first clear. The
// first person is in the list as well as in the sentence above it, because the list is
// the whole record and one that started at the second name would read as one. A single
// clear draws no list: the sentence above it already is that list.
function clearsHtml(standing) {
  const clears = standing.clearedBy || [];
  if (clears.length < 2) return '';
  return `<div class="muted small u-mt-8">${clears.length} people have
    beaten it${info('clears')}, in the order they did.
    <ol class="u-m-4-0-0-18 u-p-0">${clears.map(clear => `
      <li>${clear.handle ? personHtml(clear.handle) : 'somebody'}${
        clear.when ? ', ' + esc(clear.when.slice(0, 10)) : ''}, playing ${
        clear.seat === 0 ? 'first' : 'second'}</li>`).join('')}</ol></div>`;
}

// M5. Everyone seated in that first match is named, because the milestone is theirs
// jointly -- today one seat is always a computer player, and phase 3's two people will
// both hold it.
function firstPlayedHtml(standing) {
  const first = standing.firstPlayed;
  const who = (first && first.handles) || [];
  if (!who.length) return '';
  return `<div class="muted small u-mt-8">First played by
    <b>${who.map(handle => personHtml(handle)).join(' and ')}</b>${
    first.when ? ', ' + esc(first.when.slice(0, 10)) : ''}${info('firstPlayed')}.</div>`;
}

// S2. A live standing, which somebody else can take by playing well, and which nothing
// takes away while its holder is absent.
function highestHtml(standing) {
  const best = standing.highest;
  const who = (best && best.handles) || [];
  if (!who.length) return '';
  const many = who.length > 1;
  return `<div class="muted small u-mt-8">The highest-rated ${
    many ? 'people' : 'person'} here${info('highestPerson')} ${many ? 'are' : 'is'}
    <b>${who.map(handle => personHtml(handle)).join(' and ')}</b>${typeof best.rating === 'number'
      ? `, rated ${Math.round(best.rating)} here` : ''}.</div>`;
}

// ------------------------------------------------------- the badges this game holds
//
// Box E4 of `docs/archive/plan-meta-game.md`. A badge here is the game's own -- people
// chose it and rated it highly, and the index disagreed -- rather than the reader's. A
// game with none draws nothing: no heading, no empty slot, no greyed list, and nothing
// anywhere counts them (`docs/reference/no-dark-patterns.md`, test 5).
//
// Built out of text nodes rather than out of markup, so an award's sentence reaches the
// page as text whatever a later box puts in one. The rule sentence goes behind the (i),
// which is where every other term on this page explains itself.
//
// A score badge prints what the index said on the day it was earned, under the weights
// named beside it. The score panel further down follows its own rule and may well read
// *out of date* -- both statements are true, because the badge quotes history and the
// panel reports what can be read now.
function wireBadges(game) {
  const box = document.getElementById('gameBadges');
  if (!box) return;
  const standing = game.standing || {};
  const held = standing.badges || [];
  box.replaceChildren();
  if (!held.length) return;
  for (const badge of held) {
    const line = document.createElement('div');
    line.className = 'muted small';
    line.style.marginTop = '8px';
    const name = document.createElement('b');
    name.textContent = badge.name;
    line.append(name);
    const mark = infoNode('badge', badge.rule);
    if (mark) line.append(mark);
    line.append(document.createTextNode(
      (badge.when ? ', ' + String(badge.when).slice(0, 10) : '') + '. ' + badge.sentence
      + (badge.quoted ? ' On that day, ' + badge.quoted + '.' : '')));
    box.append(line);
  }
}

// ------------------------------------------------------------------- five stars
//
// The one number on this page that people write. `votes` keeps one row per person per
// game and replaces it when they vote again, so pressing a star is the whole act and
// there is no second state to hold (`docs/archive/plan-ux.md` A4).
//
// The mean and the count are the server's. The answer to `POST /api/vote` carries both,
// and the library's card reads them from the same query (`site/browse.py::columns`), so
// the two places show one number and the page computes neither.
//
// Absent on the local application, which has no accounts and so nobody to be. A visitor
// who has not signed in reads the mean and one sentence about adding to it, because a
// control the server would answer 401 to is worse than saying so.
function votingHtml(game) {
  const standing = game.standing;
  if (!standing) return '';
  return `<div class="panel u-mt-14" id="voting">
    <h3 class="u-mt-0">What players make of it</h3>
    ${voteBody(standing)}</div>`;
}

// Redrawn whole after a vote, so what the panel says and what the stars show cannot
// drift apart.
function voteBody(standing) {
  const mine = standing.yourVote ? standing.yourVote.stars : 0;
  const control = standing.signedIn === false ? '' : `
    <div class="row u-gap-0 u-mb-2" id="stars">${
      [1, 2, 3, 4, 5].map(n => `<button id="star${n}" data-stars="${n}"
        title="rate it ${n} out of 5" aria-pressed="${n <= mine}"
        aria-label="${n} star${n === 1 ? '' : 's'}" class="u-bg-none u-bd-0 u-p-4 u-minw-40 u-minh-40 u-fs-25 u-lh-1 u-c-act"
        >${n <= mine ? '&#9733;' : '&#9734;'}</button>`).join('')}</div>`;
  // Q503f, §3.3: whether this person's stars count toward the shortlist, said beside
  // them and never a reason to refuse the vote. The server sends it on a trial game only.
  const counts = standing.voteCounts
    ? `<div class="muted small u-mt-4" id="voteCounts">${
        esc(standing.voteCounts)}</div>` : '';
  return `${control}<div class="muted small" id="voteSaid">${voteWords(standing)}</div>${
    counts}${voteTagsHtml(standing)}`;
}

// Q599, `docs/archive/plan-return.md` §5.1: up to three of eight fixed tags, chosen by tapping,
// once there are stars to carry them. Fixed so they read the same across games, and no
// box to write in: a vote holds no text (the user, 2026-10-02). Everybody reads how many
// votes carry each tag; nobody reads who chose one.
function voteTagsHtml(standing) {
  const offered = standing.voteTags || [];
  if (!offered.length) return '';
  const counted = standing.tagCounts || [];
  const tally = counted.length
    ? `<div class="muted small u-mt-6" id="tagCounts">Tagged: ${counted.map(entry =>
        `${esc(entry.words)} &middot; ${esc(entry.count)}`).join(', ')}</div>` : '';
  if (standing.signedIn === false || !standing.yourVote) return tally;
  const chosen = new Set(standing.yourVote.tags || []);
  const full = chosen.size >= (standing.mostTags || 3);
  return `<div class="small u-mt-8">Tag it, if you like: up to ${
      esc(standing.mostTags || 3)}.</div>
    <div class="chips u-mt-4" id="voteTags">${offered.map(entry => {
      const on = chosen.has(entry.tag);
      return `<button class="chip${on ? ' on' : ''}" id="voteTag-${esc(entry.tag)}" data-tag="${esc(entry.tag)}"
        aria-pressed="${on}"${!on && full ? ' disabled' : ''}>${esc(entry.words)}</button>`;
    }).join('')}</div>${tally}`;
}

// What the panel says under the stars: this person's own vote, then everybody's. An
// unvoted game says nobody has voted rather than showing a nought, the same rule the
// card follows -- "nobody has voted" is not "everybody gave it one star".
function voteWords(standing) {
  const mean = standing.votes
    ? `&#9733; ${standing.stars.toFixed(1)} from ${standing.votes} vote${
       standing.votes === 1 ? '' : 's'}.` : '';
  if (standing.signedIn === false) {
    return mean ? `${mean} Sign in to rate it yourself.`
      : 'Nobody has rated this one yet. Sign in to be the first.';
  }
  const own = standing.yourVote
    ? `You gave it ${esc(standing.yourVote.stars)} out of 5.`
    : 'Press a star to rate it.';
  return mean ? `${own} ${mean}` : own;
}

// Five buttons by id rather than one handler over the row, because the panel is
// rebuilt after every vote and each rebuild rewires the buttons it just drew.
//
// The request carries the stars and the fixed tags' keys, and nothing else: the server
// refuses any other field (Q599).
function wireVoting(game) {
  const standing = game.standing;
  if (!standing || standing.signedIn === false) return;
  const mine = () => (standing.yourVote ? standing.yourVote.stars : 0);
  const tagsNow = () => (standing.yourVote ? standing.yourVote.tags || [] : []);
  const send = (stars, tags, button) => {
    api('/api/vote', { hash: game.hash, stars, tags }).then(answer => {
      standing.yourVote = { stars, tags: answer.tags || [] };
      standing.stars = answer.stars;
      standing.votes = answer.votes;
      standing.tagCounts = answer.tagCounts || [];
      const panel = document.getElementById('voting');
      if (panel) { panel.innerHTML = voteBody(standing); wireVoting(game); }
    }).catch(error => { paintStars(mine()); fail(error, button); });
  };
  [1, 2, 3, 4, 5].forEach(n => {
    const button = document.getElementById('star' + n);
    if (!button) return;
    // Pointing at the fourth star shows four filled, so the press is read before it is
    // made. The pressed state is the vote itself and hovering never touches it.
    button.onmouseenter = () => paintStars(n);
    button.onmouseleave = () => paintStars(mine());
    button.onclick = () => send(n, tagsNow(), button);
  });
  // The tags by id too, for the same reason as the stars.
  (standing.voteTags || []).forEach(entry => {
    const button = document.getElementById('voteTag-' + entry.tag);
    if (!button) return;
    button.onclick = () => {
      const tag = entry.tag;
      const now = tagsNow();
      const next = now.includes(tag) ? now.filter(one => one !== tag) : [...now, tag];
      button.disabled = true;
      send(mine(), next, button);
    };
  });
}

function paintStars(value) {
  [1, 2, 3, 4, 5].forEach(n => {
    const button = document.getElementById('star' + n);
    if (button) button.innerHTML = n <= value ? '&#9733;' : '&#9734;';
  });
}

// Where you have got to on this game's ladder of computer players, and what is left.
function rungHtml(mine) {
  if (!mine) return '';
  if (mine.state === 'untried') {
    return `<div class="muted small u-mt-8">You have not played a rated
      game here yet.</div>`;
  }
  const rung = mine.rung;
  const reached = rung
    ? `You have beaten ${esc(shortName(rung.agent))}${
        rung.rating ? ', rated ' + Math.round(rung.rating) + ' here at the time' : ''
      }, playing ${rung.seat === 0 ? 'first' : 'second'}.`
    : 'You have not beaten one of them yet.';
  return `<div class="muted small u-mt-8">${reached}
    ${mine.won} won, ${mine.drawn} drawn, ${mine.lost} lost over ${mine.played} rated
    game${mine.played === 1 ? '' : 's'}.</div>`;
}

// Said before the game starts rather than discovered three hundred plies into it. The
// gate's own sentences, verbatim: they name what it measured and what it concluded, and
// a paraphrase here would be a second opinion nobody asked for.
function playableWarning(game) {
  if (game.playable !== false) return '';
  return `<div class="error u-mb-10">
    <strong>A quick check says this one cannot be won.</strong>
    <ul class="u-m-6-0-0-18 u-p-0">${
      (game.playableReasons || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
    <div class="muted small u-mt-6">A few hundred games of random moves
      found no ending. You can still play it &mdash; this is about the rules, not
      about you.</div>
  </div>`;
}

// Why the outside engine is not in the picker above, when it is not. The library can
// only say whether the rules translate -- see `library._fairy` -- and the three ways of
// being absent are different news: rules with no counterpart at all will never be
// playable there whatever anybody installs; rules that translate on a machine with no
// engine want a download; and rules that translate but that the installed build plays
// differently are a fact about that build, and a newer one may lift it.
//
// It never guesses between those. `game.fairyCheck` is the verdict the roster itself
// was built from -- `opponents.fairy_check` and `fairy_stockfish_for` read one memoised
// record -- so what is said here is what decided the picker, in the verification's own
// words, with the first position the two engines disagreed about. Six committed
// rulesets land in the third case on the 2021 release, `sardines` among them, and the
// card upstairs can only say "translates" about every one of them.
//
// On the site, with no engine installed, it says nothing at all (Q576, the first-visit
// study's B2). Putting a binary on PATH is the maintainer's remedy, and four readers of
// five took the sentence for something broken; an outside engine the site does not run is
// simply not among the opponents, whichever rules a game has.
//
// Where the site has an engine, it says only what a visitor can act on (G7-G9): that
// Fairy-Stockfish can't play these rules, or plays them a little differently. The reasons,
// the position the engines disagreed on and a check not yet run are the local
// application's.
function fairyNote(game, seats) {
  if ((seats || []).some(o => o.store === 'fairy')) return '';
  const check = game.fairyCheck || {};
  if (ACCOUNTS && !check.engine) return '';
  if (ACCOUNTS) {
    const said = game.fairyTranslates === false ? "can't play these rules."
      : check.ok === false ? "plays these rules a little differently, so it isn't offered."
      : '';
    return said ? `<div class="muted small u-mt-8">Fairy-Stockfish${
      info('fairy')} ${said}</div>` : '';
  }
  let body;
  if (game.fairyTranslates === false) {
    body = `These rules have no counterpart over there (${esc((game.fairyReasons || []).join('; '))}).`;
  } else if (!check.engine) {
    body = `These rules can be written as one of its variants, but no Fairy-Stockfish is
       installed on this machine. Put one on PATH or name it in CHESS_VARIANT_SANDBOX_FAIRY_STOCKFISH
       and restart. Nothing is downloaded for you.`;
  } else if (check.ok === false) {
    const where = check.fen
      ? ` At <span class="u-ff-mono">${esc(check.fen)}</span>${
          (check.ours || []).length ? ` we have ${esc(check.ours.join(' '))}` : ''}${
          (check.ours || []).length && (check.theirs || []).length ? ' and' : ''}${
          (check.theirs || []).length ? ` they have ${esc(check.theirs.join(' '))}` : ''}.`
      : '';
    body = `These rules translate, but the engine installed here (${esc(check.engine)})
       did not play the same game, so it is not offered (${esc((check.reasons || []).join('; '))}).${
       where} A newer build may agree; the translation itself is checked against the
       current rules code.`;
  } else {
    body = `These rules translate and ${esc(check.engine)} is installed, but the check
       that offers it has not run.`;
  }
  return `<div class="muted small u-mt-8">Fairy-Stockfish${
    info('fairy')} is not offered. ${body}</div>`;
}

// One seat's art. A single-colour set -- shogi, xiangqi -- tells the two players apart
// by rotating the piece rather than recolouring it, so the rotation has to be honoured
// even when only one is shown, or half the library's pieces are drawn upside down.
function pieceArt(piece, player) {
  const icon = (piece.icons || [])[player];
  return icon
    ? `<img src="${esc(icon.href)}" alt=""${icon.rotate ? ' class="u-tf-rot180"' : ''}>`
    : `<span class="letter">${esc(piece.id)}</span>`;
}

// A move is a sentence, and the vectors it was written from are behind the (i) beside
// it. `descriptors` arrives one per move; where the two are the same string the server
// found no phrase for that rule, and a second copy of it under an (i) would say nothing.
//
// Not on the board pane, which passes an index: a piece there is a click target that
// finds the piece on the board, so an (i) inside it would fire both.
function piecesHtml(piece, options) {
  const { index, selected, ranks } = options || {};
  const rules = piece.descriptors || [];
  // The site's (i) says where the sentence comes from and carries no rule: the raw rule
  // is the maintainer's reading (A12, G3).
  const moves = (piece.moves || []).map((line, at) => `<li>${esc(line)}${
    index === undefined && rules[at] && rules[at] !== line
      ? info('moveRule', ACCOUNTS ? '' : rules[at]) : ''}</li>`);
  const notes = (piece.notes || []).map(line => `<li>${esc(line)}</li>`);
  const pick = index === undefined ? '' : ` data-type="${index}"`;
  return `<div class="piece${selected ? ' on' : ''}"${pick}>
    <div class="art">${pieceArt(piece, 0)}</div>
    <div>
      <h4>${esc(piece.name)} <span class="muted small">${esc(piece.id)}</span>${
        piece.royal ? `<span class="royal" title="the piece the game is about losing, like a king">royal</span>` : ''}${
        speedPillHtml(piece, ranks)}${restPillHtml(piece)}</h4>
      <ul>${moves.concat(notes).join('')}</ul>
    </div>
  </div>`;
}

// Schema 28 (Q943b, `docs/archive/plan-speed.md` §4.5): the speeds a game's pieces have, for each
// card's hover line; empty in a game without a speed (no piece carries one).
function pieceSpeeds(pieces) {
  return [...new Set((pieces || []).map(piece => piece.speed)
    .filter(speed => speed !== undefined && speed !== null))].sort((a, b) => a - b);
}

// A piece's speed on its card, its hover line saying what the number means (`speedLine`), or
// nothing in a game without a speed.
function speedPillHtml(piece, ranks) {
  if (piece.speed === undefined || piece.speed === null || !ranks || ranks.length < 2) return '';
  return `<span class="speed" data-speed="${esc(piece.speed)}" title="${
    esc(speedLine(piece.speed, ranks))}">speed ${esc(piece.speed)}</span>`;
}

// Schema 29 (Q983b, `docs/plan-rest.md` §5.6): a piece's rest on its card, its hover line
// saying what the number means, or nothing for a piece that never rests (the card's `rest`,
// the compiled table's; absent in a game without a rest).
function restPillHtml(piece) {
  const turns = piece && piece.rest;
  if (!turns) return '';
  const words = `rests ${turns} turn${turns === 1 ? '' : 's'} after it moves: it may not make `
    + `its side's next move${turns === 1 ? '' : 's'}, unless resting pieces are all its side can move`;
  return `<span class="speed" data-rest="${esc(turns)}" title="${esc(words)}">rests ${
    esc(turns)}</span>`;
}

// K2 (`docs/archive/plan-understanding.md` §4.2 step 2): each piece with a small board of its
// own, the piece alone on an empty board with the engine's dots (`pieceMaps`, the
// designer's diagram, worked out once on the server). The pieces that move as nothing in
// chess does come first with their boards open; a piece the server found moving as one
// of chess's does says so by that piece's name and keeps its board folded. The page
// compares nothing: `like` is the server's.
function pieceBoardsHtml(game) {
  const pieces = game.pieces || [];
  const maps = game.pieceMaps || [];
  const order = pieces.map((piece, at) => at);
  const like = at => (maps[at] && maps[at].like) || null;
  order.sort((a, b) => (like(a) ? 1 : 0) - (like(b) ? 1 : 0) || a - b);
  return order.map(at => {
    const map = maps[at];
    const row = piecesHtml(pieces[at], { ranks: pieceSpeeds(pieces) });
    if (!map || !game.opening) return row;
    const said = map.like ? `moves like a chess ${esc(map.like)}` : 'where it can move';
    return `<div class="pieceAlone" data-piece="${at}">${row}
      <details class="pieceMap"${map.like ? '' : ' open'}>
        <summary class="muted small">${said}</summary>
        <div class="pieceBoard">${boardPreview(loneBoard(Object.assign({}, game.opening,
          { hash: game.hash }), map.from, map.value), map)}</div>
        <div class="muted small">Alone on an empty board: <span class="u-c-dot"
          >&bull;</span> where it can move, <span class="u-c-bad">&#9711;</span>
          where it can take.</div>
      </details>
    </div>`;
  }).join('');
}

// Which side you take, and who takes the other one.
//
// Two anonymous dropdowns could already express this -- seat yourself second and put an
// agent first -- and never said which seat was which side, so nobody did. On a symmetric
// game that costs you the first move; on horde one seat is thirty-six pawns and the
// other is an ordinary army, and the pair of them are two different games.
//
// The armies are shown only where they differ, because a line saying both sides start
// with the same pieces is a line that has to be read to learn nothing. The names come
// from the server: "white" and "black" are what this board draws, except on a
// single-colour set where the sides are told apart by turning a piece around.
function sidePicker(game, options, table) {
  const names = game.sides || ['white', 'black'];
  const armies = game.armies || [[], []];
  const same = JSON.stringify(armies[0]) === JSON.stringify(armies[1]);
  const side = player => `
    <div class="side">
      <div class="row u-gap-7">
        <b>${esc(names[player])}</b>
        <span class="muted small">${game.rounds ? 'moves at the same time'
          : player === 0 ? 'moves first' : 'moves second'}</span>
      </div>
      ${seatCardsHtml('seat' + player, options, player === 0 ? 'human' : secondSeat(game), table,
        game)}
      <div class="muted small plan" id="plan${player}"></div>
      ${same ? '' : `<div class="muted small army">${armyHtml(armies[player])}</div>`}
    </div>`;
  return `<div class="sides">${side(0)}${side(1)}</div>` + (same ? '' :
    `<div class="muted small u-mt-8">The two sides do not start with the
      same army, so which one you take is part of the game.</div>`);
}

// Who takes a seat, as cards (F7's look, V6 of `docs/plan-visual.md`, §3.5): *you* and
// each computer player on the ladder every board shares, weakest first, the word first
// and its strength in the server's words under it (`optionStrength`, Q579's), the chosen
// one bordered in the action colour. A player every strong seat is taken from says so on
// its card in a muted line (L11), never in red. The rest of the roster -- the rules of
// thumb, the priced pieces, the outside engine -- is longer than a row of cards should be, so it
// stays in the menu, behind *more*; the menu is the seat's one value either way, and a
// card only sets it. Where the roster has no ladder at all the menu is drawn as before.
function seatCardsHtml(id, options, chosen, table, game) {
  const menu = seatSelect(id, options, chosen, table, game);
  const cards = options.filter(o => o.agent === 'human')
    .concat(options.filter(o => o.store === 'ladder' && o.agent !== 'human'));
  if (cards.length < 2) return menu;
  const others = options.length > cards.length;
  const carded = cards.some(o => o.agent === chosen);
  const busy = seatNotes(game);
  const card = o => {
    const strength = o.agent === 'human' ? 'you play this side'
      : optionStrength(o.agent, table, game).replace(/^\s*\u00b7\s*/, '').trim();
    return `<button type="button" class="seatCard" data-agent="${esc(o.agent)}"
      aria-pressed="${o.agent === chosen}"><b>${esc(o.name || o.label)}</b>${
      o.agent === gentlestOf(game) ? '<span class="gentlest">gentlest</span>' : ''}${
      strength ? `<span class="muted">${esc(strength)}</span>` : ''}${
      busy[o.agent] ? `<span class="muted">${esc(busy[o.agent])}</span>` : ''}</button>`;
  };
  return `<div class="seatCards" data-seat="${id}" role="group">${cards.map(card).join('')}${
    others ? `<button type="button" class="seatCard more" data-more="1"
      aria-pressed="${!carded}"><b>more</b><span class="muted">with a plan</span></button>` : ''}
    </div><div class="seatMenu"${carded ? ' hidden' : ''}>${menu}</div>`;
}

// A card sets its seat's menu and says so through the menu's own change, so everything
// that follows the menu (the start button's words, the plan under it) follows a card.
// *more* opens the menu for the rest of the roster.
function wireSeatCards() {
  document.querySelectorAll('.seatCards').forEach(row => {
    const menu = document.getElementById(row.dataset.seat);
    const box = row.nextElementSibling;
    if (!menu || !box) return;
    const mark = () => {
      let carded = false;
      row.querySelectorAll('[data-agent]').forEach(card => {
        const on = card.dataset.agent === menu.value;
        carded = carded || on;
        card.setAttribute('aria-pressed', String(on));
      });
      // A player the cards do not show (swapped in from the other seat) keeps its menu open.
      if (!carded) box.hidden = false;
      const more = row.querySelector('[data-more]');
      if (more) more.setAttribute('aria-pressed', String(!box.hidden));
    };
    row.querySelectorAll('[data-agent]').forEach(card => {
      card.onclick = () => {
        box.hidden = true;
        menu.value = card.dataset.agent;
        menu.dispatchEvent(new Event('change'));
      };
    });
    const more = row.querySelector('[data-more]');
    if (more) more.onclick = () => { box.hidden = false; mark(); menu.focus(); };
    menu.addEventListener('change', mark);
    // Where the row scrolls (a phone), the chosen card is brought into it.
    const chosen = row.querySelector('[aria-pressed="true"]');
    if (chosen && row.scrollWidth > row.clientWidth) row.scrollLeft = chosen.offsetLeft - 8;
  });
}

// What the opponent list says after a player's name in place of its strength, the
// server's words both: *busy* while every strong seat is taken (L11), and for a visitor,
// *sign in to play* on every player stronger than *hard* (L15, `docs/archive/plan-launch.md`
// §3.6), which says more to a visitor than *busy* does. Still choosable either way; the
// start's refusal says the rest.
function seatNotes(game) {
  const standing = ACCOUNTS && game && game.standing;
  if (!standing) return {};
  return Object.assign({}, standing.busy || {}, standing.signInFor || {});
}

// The opponent a game page offers first. On the site it is *hard* (`docs/archive/plan-launch.md`
// §5.2, L11): strong enough to be a game, never the strongest, whose places run out. The
// local application keeps *medium*, the budget it has always opened on.
//
// A person's first game on a game seats the gentlest instead (K3): `firstSeat`.
function secondSeat(game) {
  const first = firstSeat(game);
  if (first) return first;
  return ACCOUNTS ? 'minimax:nodes=6400' : 'minimax:nodes=1600';
}

// Counts per piece type, the piece's letter with its name behind it: an army summary
// wide enough to read at a glance has no room for "thirty-six pawns".
function armyHtml(army) {
  if (!army || !army.length) return 'nothing on the board';
  return army.map(piece =>
    `<span title="${esc(piece.name)}">${piece.count} ${esc(piece.id)}</span>`).join(', ');
}

// An opponent arrives as {agent, label, group, plan}. The agent string is the identity
// the ladder rates and the record stores; the label is what a person can choose from,
// because "minimax:nodes=6400" says nothing about how hard the thing will be.
//
// Grouped, because the two kinds are two different questions. A node budget asks how
// hard you want it; a rule of thumb asks what you want it to be *trying to do*, and
// they all search the same 6,400 positions so that picking between them varies one
// thing. The Elo is per ruleset for the reason the whole panel below exists: 6,400
// nodes is +770 on chess and level with random on racing kings, so a budget is not a
// strength.
//
// The seat itself comes first, then the rungs every board shares, and everything else
// under *more*: the plans, the priced pieces and the outside engine are extras, and most of them
// are not rated on a given game.
//
// **An option leads with the opponent's name, then its strength** (F7, \u00a73.3 of
// `docs/reference/plain-words.md`; the audit's P2 and P6); what it does in plain words goes under the
// menu (`planFor`), where a phone's narrow menu does not cut it off. The strength on the
// site is the one scale people are rated on, the server's numbers both: *rated 1743
// here*, or signed in and rated on this game, how far it sits from the reader, *about 240
// above you*. The local application has no people and keeps its tournament number.
function seatSelect(id, options, chosen, table, game) {
  const you = options.filter(o => o.agent === 'human');
  const rungs = options.filter(o => o.store === 'ladder' && o.agent !== 'human');
  const groups = [];
  options.filter(o => o.agent !== 'human' && o.store !== 'ladder').forEach(o => {
    const found = groups.find(g => g.name === o.group);
    (found || groups[groups.push({ name: o.group, items: [] }) - 1]).items.push(o);
  });
  // A player every strong seat is taken from is marked in the server's words before
  // anybody picks it (L11), in place of its strength: the start would be refused, and
  // the refusal should never be the first a person hears of it. Still a choice, since a
  // place can come free between the page drawing and the press.
  const busy = seatNotes(game);
  const option = o => {
    const strength = busy[o.agent] ? ` \u00b7 ${busy[o.agent]}`
      : optionStrength(o.agent, table, game);
    const gentle = o.agent === gentlestOf(game) ? ' \u00b7 gentlest' : '';
    return `<option value="${esc(o.agent)}"${o.agent === chosen ? ' selected' : ''}
      >${esc(o.name || o.label)}${esc(gentle)}${esc(strength)}</option>`;
  };
  const group = (label, items) => items.length
    ? `<optgroup label="${esc(label)}">${items.map(option).join('')}</optgroup>` : '';
  return `<select id="${id}">${you.map(option).join('')}`
    + group((rungs[0] || {}).group || 'computer players, weakest first', rungs)
    + groups.map(g => group(g.name ? `more \u2014 ${g.name}` : 'more', g.items)).join('')
    + `</select>`;
}

// What follows an option's name. Nothing is worked out here but which way round to say
// a distance the server already took (`SiteApp._standing`'s `yourGaps`, positive above
// the reader); rounding is the drawing.
function optionStrength(agent, table, game) {
  if (agent === 'human') return '';
  const standing = ACCOUNTS && game && game.standing;
  if (standing) {
    const gap = (standing.yourGaps || {})[agent];
    if (typeof gap === 'number') {
      const points = Math.round(gap);
      return points > 0 ? ` \u00b7 about ${points} above you`
        : points < 0 ? ` \u00b7 about ${-points} below you` : ' \u00b7 about equal to you';
    }
    const here = (standing.siteRatings || {})[agent];
    return typeof here === 'number' ? ` \u00b7 rated ${Math.round(here)} here` : '';
  }
  const elo = eloOf(table, agent);
  return elo === null ? '' : `  (${elo > 0 ? '+' : ''}${Math.round(elo)})`;
}

// Under the menu, what the chosen opponent is: its plain words (*Looks at 1,600 positions
// before each move.*), then a rule of thumb's plan, a sentence somebody could follow. A
// plan is the reason a heuristic carries English at all -- `agents/heuristics.Heuristic.plan`
// exists for the report and for this -- and a weight vector nobody can state is a fit,
// not a plan.
function planFor(agent) {
  const found = ROSTER.find(o => o.agent === agent);
  if (!found) return '';
  const about = found.about
    ? found.about.charAt(0).toUpperCase() + found.about.slice(1) + '.' : '';
  return [about, found.plan || ''].filter(Boolean).join(' ');
}

// Prose, and deliberately no number. Everything else on this page is a measurement;
// this is the only place that says what the game is and whether anyone would want it.
// A game with no verdict shows nothing at all rather than an empty panel -- absent is
// not the same as "judged and found wanting", and a placeholder reads as the latter.
function verdictHtml(v) {
  if (!v || !v.verdict) return '';
  const list = (label, items) => (items && items.length)
    ? `<div class="u-mt-8"><span class="muted small">${label}</span>
       <ul class="u-m-4-0-0-18 u-p-0">${
         items.map(i => `<li>${esc(i)}</li>`).join('')}</ul></div>` : '';
  return sectionHtml('verdict', 'Whether it is worth playing', `<div class="panel">
      <p class="u-mt-0">${esc(v.verdict)}</p>
      ${list('what works', v.strengths)}
      ${list('what does not', v.weaknesses)}
      ${v.suspected_degeneracy ? `<div class="u-mt-8">
        <span class="muted small">how it would most likely go wrong</span>
        <div>${esc(v.suspected_degeneracy)}</div></div>` : ''}
      <div class="muted small u-mt-10">
        A language model wrote this after reading the rules and a few games. It was
        deliberately shown none of the numbers.
        ${v.would_play_again ? 'It would play again.' : 'It would not play again.'}
      </div>
    </div>`);
}

function provenanceHtml(p, game) {
  const rows = [];
  // `html` is markup already escaped, for the one value that is a link (Q760: the
  // designer's handle, linked to their page).
  const line = (label, value, html) => value ? rows.push(
    `<tr><td class="muted u-w-110">${label}</td><td>${html || esc(value)}</td></tr>`) : null;
  // The values `provenance.origin` actually carries across the corpus, said in words. A
  // value not in this list is printed as it stands rather than dropped: the field is free
  // text by design and a spec may honestly say something nobody anticipated.
  const ORIGINS = {
    human: 'invented by a person, before this project',
    generated: 'invented by a design run here',
    evolved: 'invented by a design run here',
    designed: 'written for this project by hand',
    control: 'written for this project as a control',
    unrecorded: 'not recorded',
  };
  // The same, where the site's reader is a visitor (§3.2: a design run's game is *invented
  // by the computer*; a control is a comparison game made flawed on purpose).
  const SITE_ORIGINS = {
    generated: 'invented by the computer here',
    evolved: 'invented by the computer here',
    designed: 'written for this site by hand',
    control: 'written as a deliberately flawed comparison game',
  };
  // A game somebody built in the designer and saved (`source` *designed*). Its save is
  // filed like a design run's candidate, so the archive's lineage named a run called
  // *designed* and a round 0, and the origin read *invented by a design run here*: a
  // machine's credit on a person's game (the first-visit study's F9). It is credited to
  // its designer's handle where the site sends one (`designer.by`), and the run and the
  // round, which say nothing about a game nobody ran, are left out. The local application
  // has no accounts and says where it was made instead.
  if (game && game.source === 'designed') {
    const by = game.designer && game.designer.by;
    if (by) line('where from', `designed by ${by}`, `designed by ${personHtml(by)}`);
    else line('where from', 'built in the designer here');
  } else if (ACCOUNTS) {
    // The site names no design run and no run id: what made the game, how many rounds of
    // changes it took, and what it was asked to try (G36-G37).
    line('where from', SITE_ORIGINS[p.origin] || ORIGINS[p.origin] || p.origin);
    line('by', p.author);
    line('when', p.year);
    line('described in', p.source);
    if (p.generation !== undefined && p.generation !== null) {
      line('how', `after ${p.generation} round${Number(p.generation) === 1 ? '' : 's'} of changes`);
    }
    line('it was asked to try', p.theme);
  } else {
    line('where from', ORIGINS[p.origin] || p.origin);
    line('by', p.author);
    line('when', p.year);
    line('described in', p.source);
    line('design run', p.run);
    if (p.generation !== undefined) line('found in round', String(p.generation));
    line('the run was asked to try', p.theme);
  }
  if (p.parents && p.parents.length) {
    // Two shapes for the same fact: a design run's archive knows the parent's name, and
    // a spec's own provenance can only carry the hash, because that is all the schema
    // allows a committed file to say. The library has the names, so a bare hash is
    // looked up rather than printed.
    // A parent nobody can name is shown by its hash on a checkout, where the hash finds
    // it, and on the site as what it is to a visitor (G38): no link, since no page lists it.
    rows.push(`<tr><td class="muted">changed from</td><td>` + p.parents.map(entry => {
      const hash = typeof entry === 'string' ? entry : entry.hash;
      const named = (typeof entry === 'string' ? '' : entry.name) || nameOf(hash);
      if (!named && ACCOUNTS) return 'an earlier game no longer listed';
      return `<button class="crumb u-td-underline" data-hash="${esc(hash)}">${esc(named || hash)}</button>`;
    }).join(' &larr; ') + `</td></tr>`);
  }
  let html = rows.length ? `<table>${rows.join('')}</table>`
                         : `<span class="muted">Nothing is recorded about where this came from.</span>`;
  // The history and the notes are the maintainer's records (runs, hashes, who decided
  // what): drawn locally, never on the site, which does not send them either (G39, W7).
  if (p.history && !ACCOUNTS) html += `<p>${esc(p.history)}</p>`;
  if (p.notes && !ACCOUNTS) html += `<h3>How it played</h3><pre>${esc(p.notes)}</pre>`;
  setTimeout(() => view.querySelectorAll('[data-hash]').forEach(node => {
    if (node.tagName === 'BUTTON') node.onclick = () => go('#/game/' + node.dataset.hash);
  }), 0);
  return html;
}

// ------------------------------------------------------- what a measurement means

// The curve, drawn. Two numbers side by side -- "0.62" raw and "0.81" scored -- cannot
// say *why*, and the why is the whole content of a versioned weights file: which
// direction the curve runs, where its good band starts and stops, and how far this game
// sits from either end. A band and a ramp look different at a glance, which is the point:
// "more is better" is wrong for every component a version curves on a band, and reading
// it off the picture costs nothing.
function curveSvg(curve, raw) {
  const p = curve.points || {};
  let stops;
  if (curve.kind === 'higher_better') {
    stops = [[p.zero_at, 0], [p.good_at, 1]];
  } else if (curve.kind === 'lower_better') {
    stops = [[p.good_at, 1], [p.zero_at, 0]];
  } else {
    stops = [[p.zero_low, 0], [p.good_low, 1], [p.good_high, 1], [p.zero_high, 0]];
  }
  const xs = stops.map(s => s[0]).concat(raw === undefined || raw === null ? [] : [raw]);
  let lo = Math.min(...xs), hi = Math.max(...xs);
  const pad = (hi - lo) * 0.12 || 1;
  lo -= pad; hi += pad;
  const W = 300, H = 74, T = 8, B = 58;
  const X = v => ((v - lo) / (hi - lo)) * W;
  const Y = s => B - s * (B - T);

  const path = [[lo, stops[0][1]], ...stops, [hi, stops[stops.length - 1][1]]]
    .map(([v, s], i) => `${i ? 'L' : 'M'}${X(v).toFixed(1)},${Y(s).toFixed(1)}`).join(' ');

  let here = '';
  if (raw !== undefined && raw !== null && isFinite(raw)) {
    const x = X(raw), y = Y(scoreOn(curve, raw));
    here = `<line x1="${x.toFixed(1)}" y1="${T - 3}" x2="${x.toFixed(1)}" y2="${B}"
              stroke="var(--meter)" stroke-width="1" stroke-dasharray="2 2"/>
            <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.5" fill="var(--meter)"/>
            <text x="${Math.min(Math.max(x, 14), W - 14).toFixed(1)}" y="${H - 2}"
              text-anchor="middle" font-size="10" fill="var(--meter)">${trim(raw)}</text>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img">
    <line x1="0" y1="${B}" x2="${W}" y2="${B}" stroke="var(--line)"/>
    <line x1="0" y1="${T}" x2="${W}" y2="${T}" stroke="var(--line)" stroke-dasharray="3 3"/>
    <path d="${path}" fill="none" stroke="var(--muted)" stroke-width="1.6"/>
    ${here}
    <text x="2" y="${B + 12}" font-size="10" fill="var(--muted)">${trim(lo)}</text>
    <text x="${W - 2}" y="${B + 12}" font-size="10" fill="var(--muted)" text-anchor="end">${trim(hi)}</text>
    <text x="2" y="${T + 9}" font-size="10" fill="var(--muted)">scores 1</text>
  </svg>`;
}

// The same shape scoring.Curve applies, for placing the marker only. It is never the
// source of a number the page prints -- every value shown comes from the index file.
function scoreOn(curve, v) {
  const p = curve.points || {};
  const up = (x, a, b) => b === a ? (x >= b ? 1 : 0) : Math.max(0, Math.min(1, (x - a) / (b - a)));
  const down = (x, a, b) => b === a ? (x <= a ? 1 : 0) : Math.max(0, Math.min(1, (b - x) / (b - a)));
  if (curve.kind === 'higher_better') return up(v, p.zero_at, p.good_at);
  if (curve.kind === 'lower_better') return down(v, p.good_at, p.zero_at);
  return Math.min(up(v, p.zero_low, p.good_low), down(v, p.good_high, p.zero_high));
}

function trim(v) {
  const n = Number(v);
  if (!isFinite(n)) return '?';
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(2);
}

// The number, and everything under it. This is the densest panel on the site and the one
// most at risk of being written for the person who built it: every heading here is a
// question a reader might have, and every name a measurement goes by comes from
// `metrics/glossary.py` rather than from its identifier.
// `band` is where this score stands against every other measured game's, worked out by
// the server over the whole library and absent when the measurement is stale or missing.
// It sits directly under the number, because "0.52" says nothing until you know what the
// rest of the library scored.
// What the number above rests on: how many runs, what each of them scored, and what the
// +/- beside it is a claim about. `docs/archive/plan-four-seeds.md` §2.4 and §7.
//
// A run is one tournament from one seed. A game the library keeps is measured from four
// seeds and scored as the mean of the four; a game somebody has just measured is one run.
// The two intervals are different claims and the page makes both rather than averaging
// them into one hedge. One run's says how far that run's own score moves when its games
// are resampled, and four replays of the library put a fresh run within four of it. The
// mean's is the sampling error of the mean, combined from the four runs' own bootstraps.
// Q343 played the library from four more seeds, and the stored interval held the fresh
// mean on 110 of 130 readings, against 83.4% for an honest one
// (`docs/reference/seat-effect-size.md` §16).
function runsHtml(score) {
  const n = runCount(score.seeds);
  if (!n) return '';
  const each = Math.round((score.games || 0) / n);
  const per = each ? ` of ${each} games${n > 1 ? ' each' : ''}` : '';
  const totals = (score.runs || []).map(run =>
    run.index === null || run.index === undefined ? 'no score' : fmt(run.index));
  // Never "the mean of four runs" over a game that has no mean. Too few of its runs
  // could be read for one, and the sentence after this says so.
  const meant = score.total !== null && score.total !== undefined;
  const head = n === 1
    ? `One run${per}.`
    : (meant
        ? `The ${runsPhrase(score.seeds)}${per}: ${totals.join(', ')}.`
        : `${countWord(n).replace(/^./, c => c.toUpperCase())} runs${per}: ${
            totals.join(', ')}.`);

  // Runs that carried no total. Never averaged, never filled in from another run, and
  // never published as a nought -- so a game with too few readable runs has no number
  // rather than a low one, and this is where that is said.
  const used = score.totals_used;
  const missing = typeof used === 'number' && used < n
    ? (score.total === null || score.total === undefined
        ? ` Only ${countWord(used)} of its ${countWord(n)} runs could be scored, and a
            mean needs three, so it has no number rather than a low one.`
        : ` ${countWord(n - used).replace(/^./, c => c.toUpperCase())} of its ${
            countWord(n)} runs could not be scored, so the mean is over the other ${
            countWord(used)}.`)
    : '';

  const range = score.interval
    ? (n === 1
        ? ` Re-draw this run's own games and it lands between ${fmt(score.interval.ci_low)}
            and ${fmt(score.interval.ci_high)} nineteen times in twenty; playing the game
            again draws fresh games and usually lands within four of these &plusmn;.`
        : ` The &plusmn; is the sampling error of that mean, taken from the ${
            countWord(n)} runs' own games: re-draw them and the mean lands between ${
            fmt(score.interval.ci_low)} and ${fmt(score.interval.ci_high)} nineteen times
            in twenty. It says nothing about what a fresh set of ${countWord(n)} runs
            would score, because nobody has played ${countWord(n * 2)} seeds.`)
    : '';

  if (ACCOUNTS) return siteRunsHtml(score, n, per, totals, meant);
  return `<div class="muted small u-mt-4">${head}${missing}${range}</div>`;
}

// The same, as the site says it (G23): what the number is and what the ± is, in two short
// sentences, and each run's own score behind *more*. The local wording, the resampling
// and the nineteen times in twenty, stays local; the runs that could not be scored are
// said on the face, because a missing number is never left for a reader to find.
function siteRunsHtml(score, n, per, totals, meant) {
  const what = n === 1 ? 'One run' : `The average of ${countWord(n)} runs`;
  const range = score.interval ? ' The &plusmn; is how far chance alone could move it.' : '';
  const used = score.totals_used;
  const missing = typeof used === 'number' && used < n
    ? (meant
        ? ` ${countWord(n - used).replace(/^./, c => c.toUpperCase())} of its ${
            countWord(n)} runs could not be scored, so the average is over the other ${
            countWord(used)}.`
        : ` Only ${countWord(used)} of its ${countWord(n)} runs could be scored, and an
            average needs three, so it has no score rather than a low one.`)
    : '';
  const head = meant ? `${what}${info('run')}.${range}` : `${
    countWord(n).replace(/^./, c => c.toUpperCase())} run${n === 1 ? '' : 's'}${info('run')}.`;
  return `<div class="muted small u-mt-4">${head}${missing}
    <details><summary>more</summary>${
      n === 1 ? 'One run' : `${countWord(n).replace(/^./, c => c.toUpperCase())} runs`}${
      per}: ${totals.join(', ')}.</details></div>`;
}

// How many things the score's own version measures, and how many of those it curves on a
// band. Read off the version the score was taken under -- the guide is the glossary joined
// to that version's weights file (`library.score_guide`) -- and never written into the
// text: `v9` measured sixteen, `v10` thirteen, and a version that drops a family measures
// fewer again. Without a guide, which is a version this checkout no longer ships, the
// count is the components the score itself carries and no band is claimed.
function measuredParts(score, guide) {
  const declared = guide && guide.components;
  const parts = Object.values(declared || {});
  return {
    count: declared ? parts.length : Object.keys(score.scores || {}).length,
    bands: parts.filter(entry => entry.curve && entry.curve.kind === 'band').length,
  };
}

// A component's own roster dependence, beside its raw measurement -- or nothing, where
// the weights file this score was taken under does not carry the field at all. `v10` and
// every version before it carry it on no component, so this renders as nothing on any
// score taken under them; a version that carries it on some components and not others
// (an optional field, per component) shows it only where it is there.
function rosterRhoBadge(entry) {
  // Not on the site: it is the scoring's own check, not the game's (G28).
  if (entry.roster_rho === undefined || ACCOUNTS) return '';
  return `<span class="muted small" title="the rank correlation between two rosters'
    readings of this component's raw measurement, against the 0.50 it must clear to keep
    its weight">&rho; ${entry.roster_rho.toFixed(2)}</span>`;
}

// "Five of the thirteen have a band", or nothing where the version curves none on one.
function bandsSentence(parts) {
  if (!parts.bands) return '';
  const lead = countWord(parts.bands).replace(/^./, c => c.toUpperCase());
  return ` More is not always better: ${lead} of the ${countWord(parts.count)} ${
    parts.bands === 1 ? 'has' : 'have'} a band ${
    parts.bands === 1 ? 'it' : 'they'} should be inside rather than a direction to run in.`;
}

// How far the score rests on repeated games (Q953a, `metrics/repeats.py`): the share of
// different games beside the number, and, where the measurement stored `rests` (counting
// each game once moves the score by twice its sd or more), a sentence saying so. Stored
// numbers only, formatted; nothing here compares or decides. A provisional score shows
// neither, as it shows no band.
function repeatsPercent(share) {
  const whole = Math.round(share * 100);
  return share < 1 && whole >= 100 ? 'over 99%' : `${whole}%`;
}

function repeatsHtml(score) {
  const r = score && score.repeats;
  if (!r || r.all === null || r.all === undefined || isProvisional(score)) return '';
  const games = Number(r.games || 0).toLocaleString('en');
  const known = v => v !== null && v !== undefined;
  const bands = [];
  if (known(r.searching)) {
    bands.push(`${repeatsPercent(r.searching)} where both players think ahead`);
  }
  if (known(r.top_two)) {
    bands.push(`${repeatsPercent(r.top_two)} between the two strongest`);
  }
  const every = r.all >= 1 && [r.searching, r.top_two].every(v => !known(v) || v >= 1);
  const said = every
    ? `Every one of its ${games} games is a different game`
    : `${repeatsPercent(r.all)} of its ${games} games are different games${
      bands.length ? ` (${bands.join(', ')})` : ''}`;
  const line = `<div class="muted small u-mt-6">${said}${info('repeats')}.</div>`;
  if (!r.rests || !known(r.distinct_index)) return line;
  return `${line}
    <div class="verdict">
      <div>It rests on repeated games.</div>
      <div class="small u-fw-400 u-mt-5">The computer players played
        some of its games more than once, move for move. Counted once each, its games score
        ${fmt(r.distinct_index)} rather than ${fmt(score.total)}: the difference is at
        least twice the &plusmn;${fmt(score.sd)} the score carries.</div>
    </div>`;
}

// `trust` is the trust line (`trustHtml`, Q647), drawn right under the number and its
// band, before the parts, so *how far to trust it* is read with the number it is about.
// A provisional score (EV2) is drawn in the same place with its mark and sentence under
// the number, and with no band and no trust line: neither is read against a past value.
function scoreHtml(score, guide, band, trust) {
  // `measuringHtml` above has already said why there is no number.
  if (!score) return '';
  const g = guide || {};
  // Each part in the site's words where this is the site (`metrics/glossary.py`'s
  // `SITE_WORDS`, sent as `site`): its title, units and reading, and no *how* line, which
  // is the maintainer's (A19, A20). The local application reads the glossary as written.
  const comps = Object.fromEntries(Object.entries(g.components || {}).map(([key, entry]) =>
    [key, ACCOUNTS && entry.site ? { ...entry, ...entry.site, how: '' } : entry]));
  const howItWorks = (ACCOUNTS && g.howItWorksSite) || g.howItWorks || [];
  const parts = measuredParts(score, guide);
  const interval = score.interval
    ? ` <span class="muted">&plusmn;${fmt(score.interval.sd)}</span>`
    : '';
  // One bar per family the score carries, which is one per family its own version
  // declares: `families` is the scoring pass's, stored beside the number under that
  // version, and this page adds none and drops none.
  const families = Object.entries(score.families || {}).sort();
  const components = Object.entries(score.scores || {})
    .sort((a, b) => (score.contributions[b[0]] || 0) - (score.contributions[a[0]] || 0));
  const name = key => (comps[key] && comps[key].title) || key.replace(/_/g, ' ');
  const split = (score.violations || []).length
    ? []
    : splitSentences(score.verdict, score.runs);

  // No number, rather than a dash where a number goes. A measurement whose runs could
  // not be read is not a game that scored badly, and `runsHtml` below says which runs
  // and why.
  const scored = score.total !== null && score.total !== undefined;

  return `
    <div class="u-fs-26 u-fw-600 u-lh-1_1">${
      scored ? fmt(score.total) + interval : '<span class="muted">no score</span>'}</div>
    <div class="muted small">${scored ? 'out of 1' : 'it could not be scored'}${
      score.viable === false ? ', and ruled out' : ''
    } &middot; ${esc(rulesVersion(score.weights_version))}${info('weights')}</div>
    ${provisionalMarkHtml(score)}
    ${certifiedMarkHtml(score)}
    ${runsHtml(score)}
    ${isProvisional(score) ? '' : bandHtml(band)}
    ${trustHtml(trust, score.status)}

    ${(score.violations || []).length ? `<div class="verdict out">
        <div>It is ruled out${info('gated')}, not marked down.</div>
        <div class="small u-fw-400 u-mt-5">${
          (siteSaid(score, 'violations') || score.violations).map(esc).join('<br>')}</div></div>` : ''}

    ${split.length ? `<div class="verdict">
        <div class="small u-fw-400">${split.map(esc).join('<br>')}</div>
      </div>` : ''}

    ${seatLeanText(score) ? `<div class="verdict">
        <div>${ACCOUNTS ? 'One side wins more often'
          : 'It leans toward one seat'}, and it is ranked anyway.</div>
        <div class="small u-fw-400 u-mt-5">Over its games, ${
          esc(seatLeanText(score))}.</div>
      </div>` : ''}

    ${repeatsHtml(score)}

    <details class="u-mt-10">
      <summary class="muted small">${ACCOUNTS
        ? 'how the score is worked out' : 'how this number is arrived at'}</summary>
      <div class="why">
        ${ACCOUNTS
          ? `<p>Some tests rule a game out${info('gated')}; the score${info('index')} is the
            number above; it is made of the ${countWord(parts.count)} things measured
            below.</p>`
          : `<p>This is the project's own account of its scoring, in the same words the
            command line prints. A <b>gate</b>${info('gated')} rules a game out, the
            <b>index</b>${info('index')} is the score itself, and a <b>component</b> is one
            of the ${countWord(parts.count)} things measured.</p>`}
        ${howItWorks.map(h =>
          `<p><b>${esc(h.heading)}.</b> ${esc(h.body)}</p>`).join('')}</div>
    </details>

    <h3>What it is good at</h3>
    ${families.map(([fam, value]) => `
      <div class="fam">
        <span data-gloss="family" role="button" tabindex="0" class="u-cur-pointer u-td-underline-dotted"
          data-more="${esc((g.families && g.families[fam] || {}).means || '')}"
          title="${esc((g.families && g.families[fam] || {}).means || '')}">${esc(fam)}</span>
        <div class="bar"><div class="${styleClass(`width:${Math.round(value * 100)}%`)}"></div></div>
        <span class="num">${value.toFixed(2)}</span>
      </div>`).join('')}
    <div class="muted small u-mt-6">
      Tap or hover a name for what it asks. They are multiplied rather than averaged, so
      being near zero on one of them pulls the whole number down instead of being made up
      for elsewhere.
    </div>

    <h3>The ${countWord(parts.count)} things measured</h3>
    <div class="muted small u-mb-6">
      Each is scored out of 1, and the ones that count most come first. Beside each is
      what was actually measured. Open one for what it means.
    </div>
    ${components.map(([key, value]) => {
      const entry = comps[key] || {};
      const raw = (score.raw || {})[key];
      return `<details class="comp">
        <summary>
          <span>${esc(name(key))}</span>
          <span class="scored">${value.toFixed(2)}</span>
          <span class="meta">
            <span class="raw">${raw === undefined ? 'not measured' : trim(raw)}${
              entry.units ? ' ' + esc(entry.units) : ''}</span>
            ${rosterRhoBadge(entry)}
            <span class="bar"><div class="${styleClass(`width:${Math.round(value * 100)}%`)}"></div></span>
          </span>
        </summary>
        <div class="why">
          <p>${esc(entry.what || '')}</p>
          ${entry.how ? `<p><b>How it is measured.</b> ${esc(entry.how)}</p>` : ''}
          <p><b>How to read it.</b> ${esc(entry.reading || '')}</p>
          <p><b>It counts for</b> ${((entry.weight || 0) * 100).toFixed(1)}% of the score,
             under <b>${esc(entry.family || '')}</b>.${
             entry.optional ? ' Some games cannot supply this one, and are not marked'
               + ' down for it.' : ''}</p>
          ${entry.roster_rho === undefined || ACCOUNTS ? '' : `<p><b>Roster dependence.</b> Its raw
             measurement's rank correlation between two rosters is ${
             entry.roster_rho.toFixed(2)}, against the 0.50 a component needs to keep its
             weight.</p>`}
          ${entry.curve ? `<div class="curve">${curveSvg(entry.curve, raw)}</div>
            <p class="small">The curve shows how the measurement turns into a score out of
            1, and where this game falls on it.${bandsSentence(parts)}</p>` : ''}
        </div>
      </details>`;
    }).join('')}

    ${(g.gates || []).length ? `<h3>${ACCOUNTS ? 'Tests a game must pass'
        : 'What rules a game out entirely'}${info('gated')}</h3>
      ${ACCOUNTS ? `<div class="muted small u-mb-6">A test on the
        worse half reads whichever half of the computer players, weaker or stronger, does
        worse.</div>` : ''}
      ${g.gates.map(gate => {
        // The site's words for the test where the server sent them (`web/sitescore.py`).
        const said = ACCOUNTS && gate.site ? { ...gate, ...gate.site } : gate;
        const raw = (score.raw || {})[gate.component];
        const unit = ACCOUNTS && said.units ? ' ' + said.units : '';
        const bound = gate.minimum !== null && gate.minimum !== undefined
          ? `at least ${gate.minimum}${unit}` : `at most ${gate.maximum}${unit}`;
        return `<div class="gate"><b>${esc(said.title)}</b> must be ${esc(bound)}${
          raw === undefined ? '' : `; this one measures ${trim(raw)}${esc(unit)}`}. ${
          esc(said.reason)}</div>`;
      }).join('')}` : ''}

    ${(score.unreadable || []).length ? `<div class="muted small u-mt-10">
      ${score.unreadable.map(u => esc(name(u))).join(', ')}
      ${score.unreadable.length === 1 ? 'was' : 'were'} left out, because the ${
      ACCOUNTS ? 'computer players' : 'opponents'}
      could not play this game well enough for the measurement to be about the game rather
      than about them. It is not a mark against the game.</div>` : ''}`;
}

// ------------------------------------------------------------ proven positions
//
// R9b (`docs/plan-website.md` *Proven positions*, `site/proven.py`); since Q993b
// (`docs/archive/plan-proven.md` §2) a summary and a link. Positions from this game's own recorded
// games that the computer proved won for the side to move are worked through on a page of
// their own (`proven.js`, `/game/<hash>/proven`): this panel says how many there are, the
// reader's own progress where there is any, and the way there. Every sentence is the
// server's. The local application sends no `proven`, and draws nothing.
function provenHtml(game) {
  const proven = game.standing && game.standing.proven;
  if (!proven) return '';
  // Q993g: a record whose every position was too easy says so, rather than listing none.
  if (!proven.count) {
    if (!proven.none) return '';
    return `<div class="panel proven u-mt-14" id="proven">
    <h3 class="u-mt-0">Proven positions${info('provenPositions')}</h3>
    <p class="muted small" id="provenNone">${esc(proven.none)}</p>
    <details class="muted small u-mt-8">
      <summary>how these were proved</summary>
      <p id="provenBasis">${esc(proven.basis)}</p>
    </details>
  </div>`;
  }
  return `<div class="panel proven u-mt-14" id="proven">
    <h3 class="u-mt-0">Proven positions${info('provenPositions')}</h3>
    <p class="muted small u-m-0" id="provenSaid">${esc(proven.said)}</p>
    ${proven.yours ? `<p class="small" id="provenYours">${esc(proven.yours)}</p>` : ''}
    <div class="row u-mt-8"><a id="provenOpen" class="proven-open"
      href="${provenPath(game.hash)}">${esc(proven.open)}</a></div>
  </div>`;
}
