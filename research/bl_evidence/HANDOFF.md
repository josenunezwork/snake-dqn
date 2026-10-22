# M3 operator handoff — 2026-10-22

M3 implementation integrity: **PASS**. Both production validator entry points
now require opened, hash-bound evidence. All five negative controls were refused
through the public validators by independent Codex-V review. This admits no
production C15 clock proof, L1″ run, strict result, serving change, or release.
Existing field-only clock summaries remain blocked. B8 s3 stays the served default.

## Receipts

All paths below are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/m3-20261022/`.
JSON receipts are create-only, with SHA256 sidecars and completion markers last.

| File | SHA256 | Meaning |
| --- | --- | --- |
| `review.json` | `3f54c12cc79beb411abcdd6ac4c2faa92a2160d01f53a155841af4777157bad5` | Independent Codex-V PASS; 32 tests and five independent public-entry controls |
| `verification.json` | `92085fd9028b8735054061c5594735a44d78bd4d75d07f255a2aeda70882ea7a` | 123 passed, 12 deselected in 6.28 s; source, log, JUnit and CLI hashes |
| `reviewed.patch` | `a274405e750fa4c78dded26efc0861082af7deb2d25f030e4a34a567f52f564c` | Exact reviewed diff against `e22806354d319d61f2be9475657d2ccd746d53ab` |
| `missing-clock-check.json` | `9a6efd78640c8f1b435d0af17abf15c6fb14aec393c05c6156edc0223159c03e` | Actual CLI smoke: missing report gives NOT_READY, exit 2 |

Only this handoff document was added after the reviewed source snapshot. The
publication receipt in the same directory records the eventual commit/main SHAs.

## Ownership and changes

Sol-C was the sole writer of the three `bl_paced_identity` hooks: production
clock wrapper, receipt caller's external C binding, and explicitly non-admitting
synthetic preflight audit. M3 owns `bl_evidence`, the serving-validator hook, and
the negative controls. UI labels/recorder and controller implementation were not
changed. Existing stand-in resolver/roster tests isolate the file-reader dependency;
new public-validator controls exercise the real reader without that stub.

The reviewer reproduced a hash/read replacement race before final acceptance.
The loader now hashes the exact parsed bytes and replays a private snapshot built
from seal-pinned single-read journal payloads. The regression is in the targeted
suite. Immutable release hashes preserve the existing audited FRP3/B8 authority;
copying those pins with an assisted C is refused.

## Exact light verification commands

Run from this worktree, or from main after the published commit is present:

```sh
cd /Users/josenunez/Projects/ml/snake-dqn
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_bl_evidence.py tests/test_bl_paced_identity_receipt.py \
  tests/test_bl_paced_identity_audit.py tests/test_bl_paced_identity_clock_seam.py \
  tests/test_web_served_checkpoint.py tests/test_web_watch_roster.py \
  -k 'not TestLiveStack and not monitor and not exact_v8'
```

Check the immutable receipts and compare every reviewed source hash to main:

```sh
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
from research.bl_paced_identity.record import read_export
root = Path('/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/m3-20261022')
review = read_export(root / 'review.json')
verification = read_export(root / 'verification.json')
assert review['verdict'] == verification['verdict'] == 'PASS'
for relative, expected in review['source_sha256'].items():
    assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == expected, relative
print('M3 PASS; reviewed sources match; production qualification NOT_GRANTED')
PY
```

The exact final lint checks were Black `--check --line-length 100`, isort
`--check-only --profile black`, flake8, and `git diff --check`, restricted to
`research/bl_evidence`, the three paced-identity hook files, the serving validator,
and the four changed test files. All passed. No broad suite, timing benchmark,
training, or Colab job ran. Tests used nice 10 and one math thread.

## Remaining production prerequisites

The production provider must emit the v2 evidence contract in `README.md`,
including actual game-latch observations and operator-attested Mac execution.
Consumer-supplied frozen C/source digests are mandatory. Synthetic test corpora
prove protocol handling only. Full game/state/score replay and the later Claim B
packet remain separate requirements. Missing historical release files now fail
closed, so operators must preserve those files for normal serving resolution.

No budgets, caps, retry maxima or resume policies changed. No frozen/registered
directories, released checkpoints, serving pins, tmux sessions, or running jobs
were modified. The prohibited study directory was never opened.
