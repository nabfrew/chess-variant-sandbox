// Where a game lives on the application's page. Its router reads `#/game/<hash>`, so the
// bare `/#<hash>` these pages used to write fell through to the library and every link to
// a game opened the wrong page (`docs/archive/plan-ux.md` D0).
const GAME = '/#/game/';
