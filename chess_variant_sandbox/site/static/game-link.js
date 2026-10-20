// Where a game lives: its own address, `/game/<hash>` (G3, `docs/plan-growth.md`), which the
// server draws for a reader without script and the application's router reads. The old
// `/#/game/<hash>` still opens the game and is handed over to this address.
const GAME = '/game/';
