// An empty handle is a handle not chosen yet, and the server's answer to it is the
// format rule, which read as though the reader had typed something wrong (the
// first-visit study's N3). The format rule stays the server's, for a handle that breaks
// it; this only notices that the field is empty.
document.getElementById('go').addEventListener('click', async () => {
  say('');
  if (!document.getElementById('handle').value.trim()) {
    say('Choose a handle first: it is the name other people see beside your games.');
    return;
  }
  try {
    await post('/api/join', {
      code: document.getElementById('code').value,
      handle: document.getElementById('handle').value,
      password: document.getElementById('password').value,
    });
    window.location.href = afterwards();
  } catch (error) { say(error.message); }
});
