# E16 Mac preflight and undispatched handoff — 2026-10-21

Outcome: BUILD_VERIFIED / NOT_DISPATCHABLE. No scientific E16 efficacy rows, no Colab calls.
All new implementation is confined to research/bl_e16/. Incoming origin/main merges
retain other owners' work; no protected package was edited by this task.

## Before data

- a85d6bb2: registration and ordered kill rule frozen/pushed before allocation/gameplay.
- 940a6f98: pre-data numerical guard-A correction + E13 parity rule frozen.
- d830741d: exact prospective seed recipe frozen (main merge4d20555d).
- e8483f48: bank and freshness audit committed/pushed before Mac gameplay.
- e319c90b: parent-preserving model and ledger/statistics.
- 1fdd4862: integrated execution, disk spooling, resume, gates and transport.
- e70f4df0fb1da8ad2334c74341830de172197ca4: published manifest source pin.

PARITY_RULE.md placement differs from the judgment's suggested bl_ppo path because
this task explicitly prohibited edits there. No scientific rule was relaxed.
The original registration was preserved; its numerical paraphrase correction is a
sibling erratum, written before any allocation or gameplay.

## Allocation

Round0 CLEAR;24876distinct signed-int64 identities. Eval96/mix288total; timing3/mix;
preflight1/mix; critic4096/mix; actor4096/mix.120177explicit regularfiles scanned,
1167085historical reconstructed identities,7FRP exclusions,46alias coverage records,
zero collisions. The prohibited live study was neither opened nor traversed.
E6/E7 diagnosis: the32bit family collided repeatedly; its later high-range path also
encountered unsafe/stale inventory aliases. E16 freezes a48bit high family, retains
full collision audit, and binds published S1 replacement inventories for removed
worktree aliases. No seed substitution or bank shrinking.

BANKsha256:1a184227c8d312e6e00db53d71c994a3eddc51fd87115c40d039dc29cf7d6366
AUDITsha256:4b52ef8fd43965927c94893a71ff9112fb5d0509e0d735b050f248fb128dbca4

## Real tiny entrypoints

Interpreter:/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python.
All gameplay commands used nice-n10, SNAKE_DQN_DEVICE=cpu,
OMP/MKL/OPENBLAS/VECLIB/NUMBA_NUM_THREADS=1. Serial uses1worker; parallel uses2workers
with1maththread each. No tmux sessions, servers or training jobs were started.

`python -m research.bl_e16.runner smoke --arm A|G --bank research/bl_e16/BANK.json
--seed-audit research/bl_e16/SEED_AUDIT.json --b8-checkpoint <immutable B8 path>
--output <fresh preflight-a-01|preflight-g-01> --steps8 --rollout-steps4`

A:SMOKE_COMPLETE,1critic+7actor transitions,3updates,8.738899s wall including first
Numba compilation. G:SMOKE_COMPLETE,1critic+7actor,3updates,.618280s wall with warmed
caches. G uses7planes and the joint43→64GRU with128recordwindow. Smoke uses an explicitly
synthetic F parent around real B8 because no qualified E2F endpoint was supplied.
Its one-row truncated critic target is labellednon-scientific and cannot qualify E12.

`python -m research.bl_e16.preflight --root <artifact root> --bank research/bl_e16/BANK.json
--audit research/bl_e16/SEED_AUDIT.json --b8 <immutable B8 path> --label02`

Latest preflight-02.json:PASS;3burned natural worlds×4controllers×40frames=480active
worldsteps. Serial2.655266s,180.7729worldsteps/s; parallel3.026979s,158.5739worldsteps/s.
Deterministic rowsSHA:d1b1ebb42f844703b157b967d9a5b2dc2fbfb89d884b7197998c373fab8a955f.
Same-platform proof compares uninterrupted live continuation against a serialized
boundary restore: next3transitions, policy and Adam moment tensors identical forA/G.
Pending-critic compressed chunks/buffer/rewards/fit-offset are snapshotted; synthetic
pending-spool tests prove unchanged32row batching and complete MC targets after restore.
The initial duplicate-restore smoke remains as preflight.json;02strengthens that proof.
A transient NameError in a newly added post-step finite guard was fixed before02passed;
no scientific rows were produced by that failed verification invocation.

These H40 numbers do not qualify full-policyH5000 timing, L4 speed, E12, E13real-size
exports or platform transfer. No new scientific outcome was inspected.

## Review fixes and qualification boundaries

Independent review found and corrected: actor cut resets; incomplete critic targets;
masked-inf false tripwire; frozen actor-phase critic; mismatched parent/B8 comparator;
forged survival counters; stale imported source closure; final-step finite checks;
pending critic spool loss; weak duplicate-restore proof; tripwire label precedence.

G corpus and deployment variants must match E3/E14 canonical identities. Source pins
cover E16 plus borrowed PPO/student/simulator execution. Per-arm critic-ready.pt needs
its own SHA-bound qualification before actor unfreeze. Sidecar tests cover tampering,
partial exports, create-only refusal and unchanged supervisor discovery.

## Manifests and forecasting

Artifacts:manifest-a.json,manifest-g.json,manifest-validation.json,DISPATCH.txt under
/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e16/.
Both manifests were validated with research.colab_cli.supervisor.validate() locally,
including published commit/script,758sourcefiles,12real input payload/sidecars,
fresh absolute output, positive L4 budget,180slease, bounded export. No Colab CLI/API
call occurred. transport.dispatch_ready=false; launcher refuses before gameplay.

They are schema-valid BLOCKED templates.3minestimate/6mintimeout refer to the disabled
entrypoint refusal path; the separate scientific forecast is NOT_DISPATCHABLE.
No scientific preparation/dispatch readiness is inferred from schema validation.

Conservative literal2Mcritic+4Mactor EACH =12M forA/G;5.76Mevalworldsteps. At assumed
2000hero/s and800episodeworldsteps/s, plus2hsetup/export:5.666667h,11.333333u+4reserve.
Required2×estimate timeout680min exceeds10h/20u (22.666667u timeoutliability).
At measured4000hero/s+2000episodeworldsteps/s,3.633333h estimate/436mintimeout could fit
14.533333u liability+4reserve. Measured receipts must replace assumptions; no shrinkingN.
Shared study reservation validator sums worst-case A/G/evaluation liabilities under20u.

Open external blockers:E8-amended efficacy-unread nominated E2F; parent E12PASS;
E3GO+matchingE14PASS; measured E4 and disjoint full-policyH5000 episode ruler>=800/s;
per-arm newly refitted critic qualification; measured total study reservations and
pre-readout final endpoint lock. R/S/P remain gated stubs (not zero-valued results).
Strict/serving changes require separate Mac qualification and owner release authority.

## Final targeted verification

All commands used the primary venv, nice-n10 and one-thread CPU settings.

- `python -m pytest -q research/bl_e16 research/bl_ppo/test_model_learner.py research/bl_ppo/test_collector.py`
  =>63passed in4.95s (50E16 tests +13borrowed-backbone tests).
- `python -m pytest -q tests/test_colab_cli_supervisor.py::test_existing_output_directory_is_refused_as_nonfresh tests/test_colab_cli_supervisor.py::test_partial_local_export_without_completion_marker_is_rejected`
  =>2passed in1.10s.
- `python -m black --check --line-length100 research/bl_e16`,
  `python -m isort --profile black --check-only research/bl_e16`,
  `python -m flake8 research/bl_e16`, `bash -n research/bl_e16/colab.sh`,
  `git diff --check` =>PASS.
- Unchanged supervisor.validate loaded and validated both concrete pinned manifests.
- Two broader fake-supervisor selections were interrupted after71passes/232.01s and
  68passes+28deselected/58.05s because they include slow lifecycle timing beyond this
  transport-only change. They are not claimed as completed suites. Only their verified
  task-owned pytest PIDs were interrupted; no live jobs/tmux were touched.

## Final resume strengthening — receipt03 supersedes02

The independent follow-up required a never-restored live simulator as the control.
That stronger proof exposed NumPy pickle breaking interior/padded grid view aliases.
E16 restores those aliases locally (no shared simulator/backbone edit), with a synthetic
pickle regression test. It also finalizes a terminal critic spool before the cap may
interrupt at the final frame, so pending MC targets remain resumable at that boundary.

preflight-03.json is the authoritative latest tiny proof:PASS,480worldsteps;
serial2.772037s/173.1579steps/s; parallel3.104751s/154.6018steps/s; deterministic rowSHA
unchanged d1b1ebb42f844703b157b967d9a5b2dc2fbfb89d884b7197998c373fab8a955f.
A/G each warmed3fresh natural transitions, then an un-restored live branch and serialized
restore produced identical next3transitions, policy tensors and Adam state. Receipt02
is historical and does not supply that stronger control; it is superseded, not rewritten.
The pending critic buffer/chunk/fit-offset test and terminal-cap regression pass.
The latest full targeted selection is65tests (E16+borrowed PPO model/collector).
Use the version3manifests and DISPATCH_UPDATE_03.txt alongside the original DISPATCH.txt.
