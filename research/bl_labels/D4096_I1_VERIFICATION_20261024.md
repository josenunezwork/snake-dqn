# D4096 I1 verification

Pinned authority SHA-256:
`ccf1e0cd8d60a2019c983ab2714d58663ca42e2a228a0ddf78c16216c2ef83fb`.

The implementation was rebased onto `3488faa6`, preserving I2 and I3. The sole conflict was in the
owned `research/bl_residual/job.py`: I3's strict funding, portable discard and collection-disposition
checks were preserved, with the I1 canonical Mac discard-presence check added before input hashing.
No `d1024_resume.py` change exists. The D1024 OLD fallback remains forced to W6.

## Tests

All numerical thread environment variables were set to one. Tests were serialized without xdist.
The final launcher sets DYLD inside the niced shell and execs Python, so the macOS conftest does not
add a restart parent; CLI tests use at most the pytest process and one child Python process.

At code commit `29959ab9`, this integrated selection passed **511 tests in 55.79 s**:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SNAKE_DQN_DEVICE=cpu \
  nice -n 15 /bin/zsh -c 'export DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib; exec ./venv/bin/python -m pytest -q -o addopts= -p no:xdist research/bl_labels/test_d4096_admission.py research/bl_labels/test_d4096_resume.py research/bl_labels/test_d4096_sealed.py research/bl_labels/test_d4096_driver.py research/bl_labels/test_d4096_i3_bridge.py research/bl_labels/test_w12_runtime.py research/bl_labels/test_mac_driver.py research/bl_labels/test_w12_driver_guards.py research/bl_labels/test_d1024_resume.py research/bl_labels/test_d1024_final.py research/bl_labels/test_d1024_old_root.py research/bl_labels/test_mac_gates.py research/bl_labels/test_mac_integration.py research/bl_labels/test_w12_admission.py research/bl_labels/test_slowdown_state.py research/bl_labels/test_slowdown_resume.py research/bl_labels/test_l4_lane.py tests/test_bl_residual_pipeline.py tests/test_bl_residual_f4_fit.py tests/test_bl_residual_collection_receipts.py'
```

After the final delegation-directory persistence and inherited-priority fixes, the affected selection
passed **56 tests in 15.31 s** at code commit `71294649`:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SNAKE_DQN_DEVICE=cpu \
  nice -n 15 /bin/zsh -c 'export DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib; exec ./venv/bin/python -m pytest -q -o addopts= -p no:xdist research/bl_labels/test_d4096_resume.py research/bl_labels/test_d4096_driver.py research/bl_labels/test_l4_lane.py research/bl_labels/test_d4096_i3_bridge.py research/bl_labels/test_mac_driver.py'
```

These counts overlap; they are not added together. Earlier focused checks also passed, including
the actual I1 writer -> I3 consumer interfaces (149-test I3/receipt/pipeline selection). All initial
failures found during implementation were corrected before the passing selections above.

The final priority fix keeps D4096 workers at inherited nice >=15 rather than trying to raise their
priority to legacy nice 10 from a niced operator command. D1024 retains its default nice-10 worker
path. Synthetic fixtures cover inherited nice 15, 19 and 20 without priority escalation.

## Lint

The changed Python files passed Black (line length 100), isort (Black profile), flake8, and
`git diff --check`. The exact changed-file set is:

```text
research/bl_labels/d4096_pack.py
research/bl_labels/d4096_resume.py
research/bl_labels/d4096_sealed.py
research/bl_labels/d4096_final.py
research/bl_labels/d1024_w12.py
research/bl_labels/mac_driver.py
research/bl_labels/mac_gates.py
research/bl_labels/w12_runtime.py
research/bl_labels/l4_import.py
research/bl_labels/l4_lane.py
research/bl_labels/test_d4096_admission.py
research/bl_labels/test_d4096_resume.py
research/bl_labels/test_d4096_driver.py
research/bl_labels/test_d4096_sealed.py
research/bl_labels/test_d4096_i3_bridge.py
research/bl_labels/test_w12_runtime.py
research/bl_labels/test_l4_lane.py
research/bl_residual/job.py
```

For that set, the commands were `./venv/bin/python -m black --check --line-length 100 FILES`,
`./venv/bin/python -m isort --check-only --profile black FILES`, and
`./venv/bin/python -m flake8 FILES`, under the same nice/thread environment.
The repository source/backend gate also passed:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 nice -n 15 make PY=./venv/bin/python lint
```

Black checked 149 source/backend files; isort and flake8 passed.

## Real-pack dry run

The real start command (all paths and arguments recorded in the JSON) was run with `--check`,
nice 15 and numerical threads one. The latest immutable proof is
`ledger/D4096_I1_START_DRY_RUN_20261024_03.json`, at code commit `71294649`.

Result: **HOLD, exit 2, factory_started=false**. The B1/B2 receipt triplets do not yet exist;
the tool refuses before creating a science worktree, admission, state or output. The three canonical
admission/state/output paths were absent before and after the check. The rebound control bytes
remained unchanged. This is the expected sealed-gate refusal; it is not a claim that the live start
gate has passed. No factory was run and no production projection or funding receipt was manufactured.

## Safety and operator boundaries

No live D1024 state/output, tmux session, Mac lease, forbidden study root, label/outcome/rescue
payload, checkpoint or serving pin was opened or changed. No Colab/RunPod call, provider query or
dispatch occurred. The external writes were the authorized create-only D4096 pack and its new
binding/erratum copies. Original pack-01 and prior hash-bound evidence remain unchanged.

Independent read-only review accepted the source at `10fbb194`, then updated ACCEPT to
`71294649fcf8ad658897642250739c06db4d24bb` after reviewing the priority correction, with no
remaining P1/P2 findings. Actual B1 terminal
facts, operator attestations, B2 sealing, runtime host conditions and the operator's dispatch review
remain future gates. The exact commands are in `D4096_I1_HANDOFF.md` and were not executed.
