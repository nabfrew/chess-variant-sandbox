// Q831: ask for an invitation (`site/asking.py`). The form posts an address and an optional
// note; the server's one answer replaces the form, as text, and nothing on this page ever
// says what became of the request. A refusal (a malformed address, a long note, a full
// day) is said under the button and the form stays. The button is off while the request
// is out, so a double press sends one.
const form = document.getElementById('ask-form');
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const sending = document.getElementById('ask-send');
  say('');
  sending.disabled = true;
  try {
    const answer = await post('/api/ask-invitation', {
      email: document.getElementById('ask-email').value,
      note: document.getElementById('ask-note').value,
    });
    const done = document.createElement('p');
    done.className = 'lead';
    done.id = 'asked';
    done.textContent = answer.said;
    form.replaceChildren(done);
  } catch (error) {
    say(error.message);
    sending.disabled = false;
  }
});
