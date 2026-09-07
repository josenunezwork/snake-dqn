# State and system audit evidence — 2026-09-06

This is the small, inspectable evidence bundle for
[the expanded review](../../state_and_system_review_2026-09-06.md). The audit
examined source at `c1cf7e8c954239cb6773c3da24bd17f4ddb9b715`, based on main
`950d583b9eb0c3e316edd0f651954d133334ac1a`. It did not modify product source or
run another training campaign. The earlier sampler campaign remains separately
documented in [its result report](../../pqn_sampler_results_2026-09-06.md).

`audit_manifest.json` pins the 100 audited source/configuration files.
`completion_receipt.json` checks those hashes again at handoff.
`artifact_manifest.json` records each retained artifact's path, original path,
size and SHA-256. Scripts and original receipts are retained without rewriting
them. Absolute paths in receipts describe the machine and worktree where the
observations were made; they are provenance, not a requirement to upload data.
No model weights, replay databases or optimizer checkpoints are included.

## Evidence scope

| Directory | Retained observation |
| --- | --- |
| `vector` | Hidden boost/hunger aliases, conservative/static action-mask counterexamples, one-seed distributed-scale versus full-scale observation screen |
| `raster` | Enemy-slot paint order, wall/prediction precedence, exact observation aliases, crop and network accounting |
| `serving` | Real GameSession Play row identity, mechanics mismatch and serialized mask mismatch |
| `config` | Requested versus effective world/device, repeated model initialization with and without explicit Torch seeding |
| `pqn` | Repeated reset, per-rollout policy assignment, reward label, resume, post-floor RNG and injected numeric failure |
| `apex` | Bare replay-slot handles receiving stale priorities after overwrite |
| `dynamics` | Ordered gameplay consequences and repeated-reset parity; the receipt distinguishes rule design from contract violations |
| `evaluation` | Engine mode/config/metric differences and synthetic statistical counterexamples |
| `instrumentation` | Limits of finite-input and marginal-distribution checks |
| `verification` | Independent corroboration of four major claims and the injected nonfinite-checkpoint path |
| `prior_audit` | Previously recorded mask-escape and reflection probes, with their original source hashes; these are not new runs at the expanded-audit revision |

These are bounded counterexamples and selected test receipts. A counterexample
can disprove a universal parity claim without measuring its prevalence or effect
on a learned policy. The alias probes do not measure optimal-action disagreement.
The NaN-target probe deliberately injects a fault; the completed sampler campaign
did not produce this numeric failure. The stale replay-slot demonstration does
not measure its frequency in a full distributed run. The checkpoint-reload race
identified in the report is static conditional analysis, not a retained race run.

## Reproducing a probe

Use the audited checkout and its project dependencies (Python 3.12.3, Torch
2.9.1 on the original Mac). Most original scripts assume the layout
`<repo>/runs/<audit-directory>/<lane>/<script>.py`; do not execute the archived
copies directly from `docs/`. Stage the desired lane into a fresh ignored run
directory so generated JSON never overwrites this evidence. For example, from
the repository root:

```bash
audit_replay_dir="$(mktemp -d runs/state_system_replay_XXXXXX)"
cp -R docs/experiments/state_system_audit_2026-09-06/verification \
  "$audit_replay_dir/verification"
env PYTHONPATH=. SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  "$audit_replay_dir/verification/major_claims_probe.py" \
  > "$audit_replay_dir/verification/replayed_major_claims.stdout.log"
```

The same staging pattern works for the `vector`, `apex`, `pqn`, `serving`,
`dynamics`, `raster`, `evaluation` and `instrumentation` lanes. Some scripts
write their JSON beside themselves; others print it to stdout with a device
diagnostic prefix, so preserve that output as a log. Consult each
script or the lane receipt before invoking it. The original config seed driver
has this Mac's interpreter path embedded; use the project interpreter at that
path or adapt only the staged copy. The `prior_audit` script used a different
historical directory depth and source hash; its JSON is retained historical
evidence, not part of the above rerun recipe.

The probes use CPU with one native thread and small synthetic states. In
particular, the fault probe creates disposable synthetic checkpoints inside
its staged directory or a temporary directory; never point it at a real model.
No probe needs the incumbent champion or a training dataset. Repairing a defect
should make its old reproduction fail or change its reported observation; that
is expected and should be replaced by a regression test for the intended behavior.
