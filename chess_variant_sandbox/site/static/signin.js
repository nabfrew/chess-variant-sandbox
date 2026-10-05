async function go() {
  say('');
  try {
    await post('/api/signin', {
      handle: document.getElementById('handle').value,
      password: document.getElementById('password').value,
    });
    window.location.href = afterwards();
  } catch (error) { say(error.message); }
}
document.getElementById('go').addEventListener('click', go);
document.getElementById('password').addEventListener('keydown', (event) => {
  if (event.key === 'Enter') go();
});
