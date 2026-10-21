# Round 0 mechanics-label erratum — 2026-10-21

**The executable Round 0 world uses mechanics v2.** The v1 annotation is a
provenance-label defect, as ruled in
`docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md` §1(d).
This sibling erratum implements items (1) and (2) without changing registered
bytes or reopening any Round 0 outcome.

The incorrect labels are `profile.mechanics_version: 1` in
`research/t34_round0/BANK_ROUND0.json:82`, its hard-coded producer at
`research/t34_round0/bank.py:142`, and “mechanics v1” in
`research/t34_round0/PREREGISTRATION_ROUND0.md:46`. The constructor does not
consume the bank's `profile.mechanics_version` annotation.

## Executable configuration chain

Line references describe the unchanged source at commit
`cb9cd38d11e284bd5196999674d2e3d0a2151f23`:

1. `research/t34_round0/runner.py:330` calls `build_world(row, checkpoint)`;
   the import at `runner.py:40` selects the trajectory constructor.
2. `research/t34_exit_b8/trajectory.py:106-112` delegates to the Phase 0
   constructor. `research/phase0_search_20261019/teacher.py:101-113` resolves
   the strict context and forwards the row's seed, mix, and ordered roster to
   `bank_engine.build_world`.
3. `research/t3_rollout_labels_20261015/bank_engine.py:400-435` gets
   `grid_h5000_identity._context(1)` and calls `run_simd_eval` with
   `profile=context["profile"]`.
4. `research/redesign_scope_20261007/grid_h5000_identity.py:270-281`
   SHA-checks and loads `research/apex_safety_20260926/deployment.yaml`,
   constructs the evaluation profile, verifies its digest, and verifies the
   frozen checkpoint identities. The YAML specifies `game.mechanics_version: 2`
   at **line 33** (line 32 opens `game:`).
5. `research/apex_safety_20260926/dev_screen.py:93-94` pins deployment SHA-256
   `4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5`
   and profile digest
   `d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b`.
6. `src/simd_env/eval_engine.py:453-469` takes `mechanics_version` from
   `profile.world`. `src/evaluation/protocol.py:106-107` rejects the
   `promotion-v2-watch-rect` profile unless the arena is rectangular and
   `world.mechanics_version == 2`.

The registration **hash-binds deployment.yaml**:
`research/t34_round0/REGISTRATION_ROUND0.json:958` records the deployment
SHA-256 above in its top-level `source_hashes`; line 42 repeats it in the
preflight source closure. The same registration also binds the mislabelled bank,
preregistration, bank producer, constructor chain, and mechanics constants.
Thus the executed configuration is registered despite the incorrect prose label.

## Excluded-fixture attestation

The separate script `research/t34_round0_erratum/attest_mechanics.py` calls the
same `runner.build_world` constructor on exactly one row from
`BANK_ROUND0.json`'s `excluded_preflight_rows`: scripted fixture
`t34-round0-preflight-20261020/v1/scripted/0000`, seed **2102588111**.
It checks the registered bank digest, verifies the seed is absent from all 240
registered rows, and checks the constructor/profile files against registration
hashes before construction. Three ordinary bare-policy steps provide a bounded
smoke; no paired comparison or scientific endpoint is computed.

Run from this worktree with the project venv:

```sh
nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -B -m research.t34_round0_erratum.attest_mechanics
```

The create-only receipt is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-erratum-20261021/attestation.json`.
**Observed result: `ATTESTED_MECHANICS_V2`; one world, three frames.** Receipt
SHA-256: `a27aa16497ef9713a62a5cbf214396ab1453f1f075e5f96be78638c4593dcd80`.
It records `sim.cfg.mechanics_version == 2`, the complete `asdict(sim.cfg)`,
deployment/profile digests, registered source hashes, checkpoint identity,
fixture row, frame count, CPU thread settings, nice value, and script/source
identity. An audit hook rejects opens and directory scans of the registered
study artifact root. Execution uses Mac CPU, `nice -n 10`, and one native math
thread (the inherited nice value 5 makes the observed process nice value 15).

This attestation changes no bank, look, statistic, controller, candidate,
threshold, or serving identity. Both arms retain the same paired v2 world.
All pre-existing registered files under `research/t34_round0/` remain unchanged;
`research/t34_round0d/` remains untouched under its stage-1 freeze. No Colab,
registered-bank gameplay, study-output access, analysis, or release is involved.
