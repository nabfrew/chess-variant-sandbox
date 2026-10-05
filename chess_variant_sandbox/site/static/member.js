// R1, `docs/plan-return.md` §6.2: *Challenge ‹handle›* on a member's page.
//
// The page is drawn by the server; this opens the sheet under the button and sends the
// challenge by handle through Q571's own route, `/api/challenge`. Nothing counts down: the
// challenge waits until it is answered or withdrawn (test 2), and the site sends nothing
// but the challenge itself (test 3).
const { handle } = JSON.parse(document.getElementById('page-data').textContent);

const opener = document.getElementById('challenge-open');
const sheet = document.getElementById('challenge-sheet');
const send = document.getElementById('challenge-send');
const cancel = document.getElementById('challenge-cancel');

if (opener && sheet && send && cancel) {
  opener.addEventListener('click', () => { sheet.hidden = false; opener.hidden = true; });
  cancel.addEventListener('click', () => { sheet.hidden = true; opener.hidden = false; });
  send.addEventListener('click', async () => {
    send.disabled = true;
    const body = {
      hash: document.getElementById('challenge-game').value,
      seat: document.getElementById('challenge-seat').value,
      handle,
    };
    try {
      await post('/api/challenge', body);
    } catch (error) {
      send.disabled = false;
      say(error.message);
      return;
    }
    sheet.hidden = true;
    const note = document.getElementById('note');
    note.className = 'note done';
    note.textContent = 'Challenge sent. It waits on your page, under Challenges you sent, '
      + 'until ' + handle + ' accepts or declines it.';
  });
}
