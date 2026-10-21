// Design runs, watched while they go.

// ------------------------------------------------------------------ design runs

function showRun(run) {
  stopPolling();
  RUN = run;
  view.innerHTML = `
    <button class="crumb" id="back">&larr; all games</button>
    <h2 class="u-mt-10">Trying to improve ${esc(run.parent)}</h2>
    <div class="muted small">${esc(run.theme || 'no particular angle asked for')}
      &middot; ${run.generations} round${run.generations === 1 ? '' : 's'}
      &middot; saving to ${esc(run.directory)}</div>
    <div class="muted small u-mt-6 u-maxw-74ch">
      Each round asks a model for variations, throws out the ones that obviously do not
      work, plays a few hundred games of each survivor and scores them. Anything it keeps
      turns up in the library as a game nobody has committed to yet. You can leave this
      page; it carries on.
    </div>
    <div class="panel u-mt-14"><pre class="log" id="log"></pre></div>`;
  document.getElementById('back').onclick = () => go('#/');

  const paint = data => {
    RUN = data;
    const log = document.getElementById('log');
    if (!log) return stopPolling();
    log.textContent = data.log.join('\n');
    log.scrollTop = log.scrollHeight;
    document.getElementById('runs').textContent =
      data.state === 'running' ? 'a model is inventing variations' : `that run ${data.state}`;
    if (data.state !== 'running') stopPolling();
  };
  paint(run);
  TIMER = setInterval(() => api('/api/run/' + run.id).then(paint).catch(() => stopPolling()), 2000);
}

// A session is handed to the router rather than drawn, so starting a game pushes a
// history entry and Back returns to the ruleset it was started from.
function open_(snapshot) { SESSION = snapshot; go('#/play/' + snapshot.id); }
