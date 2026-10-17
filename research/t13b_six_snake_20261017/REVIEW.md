# T13b pre-run review

Decision rule committed/pushed first: f36a20d30f5c53cff3d22978cceafab435666b4b.
Independent read-only reviewer found no remaining blocking issues after checking:

- Exactly four six-snake rosters and one eight-snake all-B8 supplemental cell;
  25 paired seeds per cell, H5000, CPU, four children plus one coordinator.
- Six-snake direct default construction; eight-only local configuration hook;
  correct B8/FRP3/champion identities and hero/opponent wrapper scope.
- Reused immutable T13 accounting and metric_samples implementation, paired
  seed bootstrap with 10,000 draws at RNG seed 20261017.
- T7/T12/T13 registered seed exclusion and visible JSON/JSONL collision audit.
- Exit, digest, count, horizon, initial world/RNG and accounting validation.
- Source hashing includes reused research modules; output refuses overwrite.
- No tmux or external-process management; serving defaults remain unchanged.

Root and reviewer caught an intermediate nested-constructor count override and
an eight-path import issue before any measurement. Both were corrected before
smoke. The preregistration record will remain immutable once run-intent-bound;
results go in a sibling file.

Builder verification: tests/test_t13b_six_snake.py 4 passed; sequential H100 smoke
for all five cells passed, effective counts 6/6/6/6/8 and mixed opponent IDs 1–5.
Focused formatter/import checks passed. Root repeats relevant tests before launch.

Remaining gates: fixed full run, raw-evidence and CI verification, rule application,
Watch8 measured monitor constant and boundary tests, final erratum and publication.
