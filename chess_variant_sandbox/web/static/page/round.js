// The monthly round (Q503h, `docs/archive/plan-game-pipeline.md` §3.3): `#/round` for the latest,
// `#/round/<date>` for one by the 1st of its month.
//
// Everything on it is the server's (`site/rounds.py`): which games entered, who may vote on
// which and why not, and -- only after the close -- the yes and no and what the maintainer
// decided. The page counts nothing and never sends a running tally, because it is never
// sent one. Every string from the store is escaped (`esc`), names and reasons included.
//
// The local application holds no rounds; a checkout says so rather than asking.

async function showRound(day) {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  if (!ACCOUNTS) {
    view.innerHTML = `<div class="panel"><h2 class="u-mt-0">The round</h2>
      <div class="muted">Rounds are held on the site, where members vote on the shortlist.
        A checkout has no members, so it has no round.</div></div>`;
    return;
  }
  pending('Reading the round…');
  const data = await api('/api/round' + (day ? '/' + encodeURIComponent(day) : ''));
  clearPending();
  paintRound(data);
}

function paintRound(data, said) {
  const title = data.round ? 'The round of ' + data.words : 'The round';
  document.getElementById('tagline').textContent = data.round
    ? `${data.entrants.length} ${data.entrants.length === 1 ? 'game' : 'games'} in it`
    : 'members vote on the shortlist once a month';
  const others = (data.rounds || []).filter(one => one.round !== data.round).map(one =>
    `<a href="#/round/${esc(one.round)}">${esc(one.words)}</a>`).join(' · ');
  view.innerHTML = `
    <div class="panel" id="round" data-round="${esc(data.round || '')}">
      <h2 class="u-mt-0">${esc(title)}</h2>
      <div>${esc(data.says)}</div>
      <div class="muted small u-mt-6">${esc(data.rule)}</div>
      ${said ? `<div class="small u-mt-6" id="roundSaid">${esc(said)}</div>` : ''}
      <div id="roundEntrants" class="u-mt-12">
        ${data.entrants.map(entrantHtml).join('')}</div>
      ${others ? `<div class="muted small u-mt-12">Other rounds: ${others}</div>` : ''}
    </div>`;
  wireRoundVotes(data);
}

// One entrant: its four-run score, its stars and players, then either the yes and no (to
// a member who may vote), the reason they may not, or after the close its counts and fate.
function entrantHtml(one) {
  const score = one.score === null || one.score === undefined
    ? 'no current score' : `${fmt(one.score)}, the average of four runs`;
  const stars = one.stars === null || one.stars === undefined
    ? 'no stars yet' : `&#9733; ${esc(one.stars)} (${esc(one.starVotes)})`;
  const counted = one.yes === null || one.yes === undefined
    ? '' : `<div class="small">${esc(one.yes)} yes, ${esc(one.no)} no${
      one.passed ? ': passed' : ''}</div>`;
  const carried = one.carried
    ? `<div class="muted small">Carried from the round of ${esc(one.carried)}, its vote kept.</div>`
    : '';
  let vote = '';
  if (one.mayVote) {
    const mine = one.yourVote;
    vote = `<div class="row u-mt-6">
        <button class="${mine === true ? 'primary' : ''}" data-vote="yes" data-hash="${
          esc(one.hash)}">yes</button>
        <button class="${mine === false ? 'primary' : ''}" data-vote="no" data-hash="${
          esc(one.hash)}">no</button>
        ${mine === true || mine === false
          ? `<span class="muted small">You voted ${mine ? 'yes' : 'no'}; you can change it
            until the close.</span>` : ''}</div>`;
  } else if (one.voting && one.why) {
    vote = `<div class="muted small u-mt-6">${esc(one.why)}
      <a href="/game/${esc(one.hash)}">Play it</a>.</div>`;
  }
  return `<div class="entrant u-mb-12" data-hash="${esc(one.hash)}">
      <div><a href="/game/${esc(one.hash)}"><strong>${esc(one.name)}</strong></a>
        <span class="muted small">by ${personHtml(one.designer)}</span></div>
      <div class="small">${score} · ${stars} · ${esc(one.players)} ${
        one.players === 1 ? 'player' : 'players'}</div>
      ${carried}${counted}
      ${one.decisionSays ? `<div class="small">${esc(one.decisionSays)}</div>` : ''}
      ${vote}</div>`;
}

function wireRoundVotes(data) {
  document.querySelectorAll('#roundEntrants button[data-vote]').forEach(button => {
    button.addEventListener('click', () => {
      button.disabled = true;
      api('/api/round/' + data.round + '/vote',
        { hash: button.dataset.hash, yes: button.dataset.vote === 'yes' })
        .then(answer => paintRound(answer, answer.said))
        .catch(error => { button.disabled = false; fail(error, button); });
    });
  });
}
