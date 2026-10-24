# D1024 OLD W6 mechanical continuation

Status: **PREPARED_CHECKED_NOT_DISPATCHED**. Operational code pin:
`7b7b4ca2cd231bc088097c71630971cb9bf30902`, published to origin/main.
Independent Codex-V LIGHT: PASS, no findings. No new scientific look.

`RUN_D1024_RESUME.sh` and `d1024_resume.py` continue the original series after
verified operational preemptions. They do not rerun `RUN_D1024.sh`, re-admit a
control, reset accounting, or choose new scientific inputs. D4096 remains HOLD.

The prior dispatch `bl-f0-d1024-old-eq02-rebind-01` ran operational pin
`587ff73ecdb9ebb25a104c3e45450ceb851c0358`. Its terminal status is PREEMPTED,
invalid reason and registered scientific stop are both null, and its stop reason
is `operational guard: persistent concurrent compute pid 22350`. This is an
operational continuation, not a retry of a scientific failure.

The original package remains:

```text
/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01
```

- Admitted control: `d1024-admitted.json`, SHA256
  `aa8a38bc0205619ade38b24ef665ecfffcc70d7d62b82a61d23c868a40bd6d47`.
- Same state: `d1024-state-mac-01`; original cumulative wall cap **144,000 s**.
- Prior elapsed charge: **7,584.940389083 s**; worker charge:
  **45,509.642334498 worker-seconds**; remaining cap **136,415.059610917 s**.
- Science/admission pin `111c47818a097c57451386ef4f8a7009616f8f77`;
  actual OLD collector `61900af28932e47e22e332840fc4406559adb062`, **W6**.
  T-EQ equiv-02 FAIL and the calibration binding remain unchanged.
- Next output: `d1024-output-mac-02` (absent after both checks).
- Published union: `d1024-resume-02/verified-ledger.json`, SHA256
  `246bc696b61a16855660a07db9b6bd30ce9bf9cde3f8ec9bd78d093e5d159acf`.
  It contains **362 unique units: 300 lives and 62 completed sessions**.
- **2,098 unfinished parents** will be scheduled. The old terminal's 2,092
  pending parents excluded six active parents at preemption. Their completed
  life units remain reusable; they are included in the continuation.

Every invocation validates all durable start/terminal/export triplets, cumulative
wall and worker charges, output sequence, cap and control identities, and hashes
sealed rate files without opening their outcomes. Missing terminals, INVALID or
COMPLETE attempts, registered stops, orphan receipts, uncharged outputs, exhausted
caps, and omitted/duplicate/foreign units return HOLD. The original admission,
equivalence, calibration, F4 predecessor, checkpoint bytes and frozen collector
are revalidated before checking the coordinator. Check mode publishes or verifies
the same create-only ledger; it creates no output or start. After a further
verified preemption, the wrapper verifies the full chain and automatically selects
resume/output 03, then 04, and so on under the same cumulative cap.

Verification was run at nice 15 with one thread per library and no pytest workers:

```bash
nice -n 15 env PYTHONDONTWRITEBYTECODE=1 SNAKE_DQN_DEVICE=cpu \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest \
  -q -p no:xdist -p no:cacheprovider \
  research/bl_labels/test_d1024_resume.py \
  research/bl_labels/test_d256_resume.py \
  research/bl_labels/test_mac_driver.py \
  research/bl_labels/test_mac_gates.py \
  research/bl_labels/test_mac_integration.py \
  research/bl_labels/test_d1024_old_root.py \
  research/bl_labels/test_frozen_factory.py
```

Result: **91 passed in 24.58 s**. Independent reviewer: the same single-thread
nice-15 command restricted to `test_d1024_resume.py`: **41 passed in 3.14 s**.
The initial synthetic integration test needed its inline worker's F0 environment
restored before its third call; the production worker is a separate process.
The corrected fixture passes. No shared driver or collector code was changed.

These checks also passed:

```bash
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black \
  --check --line-length 100 research/bl_labels/d1024_resume.py research/bl_labels/test_d1024_resume.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort \
  --check-only --profile black research/bl_labels/d1024_resume.py research/bl_labels/test_d1024_resume.py
nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 \
  research/bl_labels/d1024_resume.py research/bl_labels/test_d1024_resume.py
/bin/bash -n research/bl_labels/RUN_D1024_RESUME.sh
git diff --check
```

The real synchronous shell entrypoint `--check` passed **twice** from the clean
published operational pin, using the environment below (invoked directly with
`nice -n 15 /usr/bin/env ... /bin/bash research/bl_labels/RUN_D1024_RESUME.sh --check`,
not through dispatch). The second check reused the sealed union unchanged. Both
returned CHECKED, workers=6, next output02, with no collection. All **1,126 original
files / 1,051,727,983 bytes** were hashed before preparation and verified unchanged.
Inventory: sibling `d1024-resume-verification-20261024-01/original-inventory.json`.
Review receipt: `CODEX_V_D1024_RESUME_20261024.json` plus SHA/completion sidecars.

Use Bash for these exact operator commands. Names were checked fresh during
preparation. Dispatching a check is asynchronous: wait for CHECKED and EXIT=0 and
for its worktree lock to be released before dispatching collection. If a name has
been consumed, choose a fresh name; do not reuse a log or runner.

```bash
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f4fix
OPS_PIN=7b7b4ca2cd231bc088097c71630971cb9bf30902
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts
D1024_ENV=(
  PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
  PYTHONDONTWRITEBYTECODE=1 SNAKE_DQN_DEVICE=cpu
  F0_COMMIT=111c47818a097c57451386ef4f8a7009616f8f77
  F1_COMMIT=111c47818a097c57451386ef4f8a7009616f8f77
  PACK="$ART/bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01"
  EQ="$ART/bl-labels-f0-speed/equiv-02/receipt.json"
  CAL="$ART/bl-labels-f0/w12-admission-01/receipt.json"
  OLD_FACTORY_ROOT="$ART/bl-labels-f0-speed/equiv-pack-02/old-reference"
  F4_INPUTS="$ART/big-lever/exp/f4/d1024-metadata-staging-03/inputs"
  F4_CONTROL="$ART/big-lever/exp/f4/d1024-metadata-staging-03/inputs/control.json"
  B8=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
  FRP3=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth
)

"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-d1024-resume-check-old-eq02-01 "$OPS_PIN" -- \
  /usr/bin/env "${D1024_ENV[@]}" /bin/bash research/bl_labels/RUN_D1024_RESUME.sh --check

# Only after CHECKED, EXIT=0, and release of the check's worktree lock:
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-d1024-resume-old-eq02-01 "$OPS_PIN" -- \
  /usr/bin/env "${D1024_ENV[@]}" /bin/bash research/bl_labels/RUN_D1024_RESUME.sh
```

Check log:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/mac-bl-f0-d1024-resume-check-old-eq02-01-7b7b4ca2cd23.log`.
Runtime priority/persistence, lease, timing, power/lid, thermal, and cumulative-cap
guards remain active at actual admission. No dispatch, workers, timing measurement,
scientific aggregation, or outcome printing occurred. No tmux/session mutation,
released checkpoint or serving-pin change, or existing hash-bound file edit occurred.
Observed sessions left untouched: `git-sync`, `job-r30-1-rehearsal-c1bf44c8-01`,
`job-r32-cpu-canary-c1ab2750-02`, `renicer`, `sol-d1024resume`, `sol-v8fastpath`, `watchd`.
