# O1 harvest-branch census

Prepared implementation of the O1 mechanism screen adopted in
[the independent judgment](../../docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md).
The 48-world census has **not run**. No Colab calls or dispatch were made.

Read [PREREGISTRATION_O1.md](PREREGISTRATION_O1.md), then
[REGISTRATION_O1_TIMING_V2.json](REGISTRATION_O1_TIMING_V2.json). The original
[REGISTRATION_O1.json](REGISTRATION_O1.json) remains preserved for the earlier
source version; the [timing erratum](PREFLIGHT_TIMING_ERRATUM.md) explains the
source-only correction and both excluded fixture attempts. The frozen bank contains 48 fresh
study identities (16/mix), two excluded L4 timing worlds, two excluded Mac
preflight worlds, and three excluded observer-parity worlds. The original
full-history audit is supplemented by a post-main-merge audit of published
inventories, including reconstructed hash-only Claim A namespace seeds.

`runner.py` follows bare B8 once per world, samples the frozen root schedule,
and evaluates all advisory-legal first actions with reacting exact cloned
opponents and H500 B8 continuation. `provenance.py` observes ambient, death-drop
and boost-pellet intake without modifying the dynamics. `analyze.py` checks
identity, accounting, full bank completion, timing admission and three-world
observer parity before its create-only one-look decision. Local branch gains
are mechanism evidence, never H5000 efficacy or release evidence.

The single L4 manifest uses the qualified supervisor with an 8-unit /240-minute
outer cap, 1.6-unit reserve, 11 one-thread CPU workers and a 64-MiB compressed
immutable export. The runtime first measures two excluded worlds/two roots each
under its <=1-unit timing cap. It refuses study-bank admission if the measured
forecast does not fit the remaining ordinary allowance. Actual charged setup is
subtracted using the supervisor's session clock. Final evidence is bundled with
the supervisor's required checksum/completion sidecars. The draft manifest is
prepared locally; the operator must supply the current reviewed account-wide
assignment inventory/rates at any later admission (three assignments maximum).

Reproduce targeted verification from this worktree, using the shared venv:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 0 \
  research/t34_o1_harvest \
  tests/test_colab_cli_supervisor.py::test_existing_output_directory_is_refused_as_nonfresh \
  tests/test_t34_transport_qualification.py::test_record_matches_actual_accepted_source
```

The tests are synthetic and excluded from scientific inference. The spawn test
proves seeded task transport/order; it does not substitute for real simulation
observer parity. The three full H5000 parity worlds remain a mandatory runtime
admission gate. Mac/x86 identity and serving qualification are outside O1.

Source and registration are published before the mandatory excluded Mac fixture.
See `VERIFICATION_O1.md` for the frozen SHAs, real entrypoint command, measured
preflight timing, extrapolated L4 forecast, receipt hashes and draft manifest.
All code changes are within this package. No frozen Round 0, Path B or ExIt source
was edited, no running tmux service was touched, and no released checkpoint or
serving pin was changed. The sealed Round 0 study directory was never opened.
