# T4e return bank and prepared launch — 2026-10-16

The return bank is complete. Collection exited 0, and both banks passed the
collector's final manifest, shard, and `ReturnBank` loader validation. Training
has **not** started. This artifact is collection evidence, not an arm result or
promotion decision.

The immutable pre-registration is
[`specs/t4e_prereg_2026-10-16.md`](specs/t4e_prereg_2026-10-16.md), SHA256
`d404e5d95e3671d12b0c0761246862448b0f84a1fbc7ffecc3046ac1b1fdd6a0`.
It was committed and pushed before collection. The separate return Huber delta
and its default-path parity tests preceded pre-registration. RETURN uses
`--return-huber-delta 100`, with TD delta 1, return weight 1, return fraction 0.5
(128 additional branch rows per update), and the full on-policy M6 CTRL anchor.

## Completed collection

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4e-return-bank-20261016`.

| Bank family | Unique states | Action branches |
| --- | ---: | ---: |
| Training decisive | 9,462 | 49,653 |
| Training matched outside | 9,462 | 48,193 |
| Held-out decisive | 1,052 | 5,418 |
| Total | 19,976 | 103,264 |

The training bank is `banks/train` (18,924 states, 97,846 branches, 330 shards).
The held-out bank is `banks/holdout` (1,052 states, 5,418 branches, 30 shards).
Both paths are relative to the artifact root above. Every retained state has
its actual all-legal-action D300 outcomes, including fatal branches; the outside
branches were replayed after exact matching and observation-identity checks.

Six complete balanced cycles played 1,200 fresh source worlds, 240 per B8 final.
Each final's cycle retained the historical 27 scripted / 9 mixed / 4 frozen
schedule. Exactly 120 worlds were assigned to holdout before play: 10% of worlds
and a realized 10.0057% of decisive states. Training excludes all holdout worlds.
Every fixed historical length-bucket × Q-decile quota matched exactly, with no
missing stratum or substitution. The completion receipt records the full counts
by final and mix and the exact matching quotas.

Outside selection followed the preregistered fixed plan/decision order after
global observation deduplication. It is not balanced by source final: B8_s0
contributes 8,571 outside states, B8_s1 518, B8_s2 373, and B8_s3/B8_s4 zero.
All five finals contribute played worlds and decisive states. This source
imbalance is a sampling limitation of the registered selection order; it was
not adjusted after observing the bank.

The task-owned `t4e-collect` session ran 12 single-thread CPU workers on AC power.
The successful invocation started at 17:49:12 UTC and finished at 19:49:29 UTC.
Measured collector wall time was **7,216.263 seconds (2 h 00 m 16.263 s)**,
including collection audits, outside replay, publication, and validation.
No training or evaluation worlds were launched. No other tmux session or
process was stopped, changed, or reused.

## Provenance and operational evidence

Collection source commit: `f134914535d4139ee56e05c854b69cfe3a6157fd`.
Source, pre-registration, reference-bank, and B8 checkpoint hashes were bound
before play and rechecked before completion. The bound source stayed unchanged
through the successful run. Every cycle has its own fresh-seed plan and audit
under `plans/`; rounds 68 and 69 were excluded from collection allocation.

The post-collection round-68/69 audit passed with no recorded/generated seed
collisions. Receipt: `t4e-validation-20261016/seed_audit_postcollection.json`,
SHA256 `bca5a5522c54120bd338a5659733e8f7af8dec2fa28a8af772e20678800cd3cc`.
It preserves exact scan-time identities for the two earlier audit receipts and
this task's execution transcript, excluded because they document the reserved
rosters rather than played worlds. Collection artifacts remained in scope.
As preregistered, refresh this audit again when evaluation is actually launched.

| Artifact (relative to bank root) | SHA256 |
| --- | --- |
| `binding.json` | `ad53a1814c26229521da406cc261d1c5f468cbc651e8f8ae8a4a22405de86f51` |
| `completion.json` | `a4437243c7899de2ab6498c3d6b27efc5179e96ec784e2afde3b631e2acab089` |
| `publication.json` | `bafb976998bdf4f83cef4087cef4fcaac41f39e6713e37c3ca29be572eabf2b3` |
| `banks/train/manifest.json` | `708e84d9463ebb890edeb3aa2ed99111cfbd259c939a3c2c211d6bf41c236a3d` |
| `banks/holdout/manifest.json` | `c0ae22541d20927d1a7996f1a5a91c2aceb9591de8eb743c514def86e2c95e6c` |

Operational receipts are in sibling directory `t4e-validation-20261016`:
`owned-session.json`, `collect-console.log`, and `collect-exit.txt` (0).
`collection-exit-verification.json` records that the exact owned tmux session
and shell PID were absent after natural exit; no stop/kill command was used.
The earlier preflight-only failure played zero worlds and remains preserved in
`t4e-return-bank-20261016-attempt1-preflight-failed` and the validation directory's
`attempt1-preflight-failed/`. Its repair and real scanner integration test are
recorded in `t4e_collection_preflight_repair_2026-10-16.json`.

## Prepared execution

After bank completion, prepared and validated:

- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4e-mac-20261016/lane0.sh`
  for RETURN seeds 10, 12, and 14.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4e-mac-20261016/lane1.sh`
  for RETURN seeds 11 and 13.

Both launchers run from `/Users/josenunez/Projects/ml/snake-dqn-t4d`, use
`--device mps`, preserve T4a's environment caps without adding further training
restrictions, and use the completed training bank only. Their `--check` mode
validates bound inputs and prints full commands without starting training.
The corresponding B8_s0..4 finals initialize the five runs; existing M6 CTRL
finals remain the control. The pre-registration fixes the fresh-round screen,
diagnostic readouts, and post-result fork.

The tracked copies are in `research/t4e_return_20261016/`. `launch_inputs.json`
binds 25 input/source files, including the return-bank target implementation,
both published manifests, the completion receipt, and the B8/CTRL checkpoints.
The preflight also verifies the exact bank shard sets and every shard hash.
Both actual artifact scripts passed `--check`, printing all five complete
commands. The capture is `t4e-validation-20261016/launch-checks-final.json`, SHA256
`472b142cb62ecf06b3c09ae43ac700e3a93d0b937e86952ef1621061207cdb9f`.
No `RETURN_s*` run directory or log exists.

Four focused launcher tests passed, including a stub-invocation check proving
that `--check` invokes only the input verifier, input-tamper and extra-shard
rejection, and existing-output refusal. Scoped flake8, shell syntax, and diff
checks passed. Earlier delta/default-parity, collection, and scanner-integration
tests are recorded in `t4e_precollection_validation_2026-10-16.json`,
`t4e_default_trace_2026-10-16.json`, and
`t4e_collection_preflight_repair_2026-10-16.json`.

The independent saved-bank audit passed all ten checks: exact hashes and counts,
world/mix allocation, all-legal-action completeness, decisive fatal/surviving
outcomes, terminal data, canonical split/family disjointness, reference-observation
exclusion, and bound provenance. Receipt:
`t4e-validation-20261016/bank-independent-verification.json`, SHA256
`e76279ceb368ac31b6c405b59b8d1c6b0749466efdb6f99a77a56b71094a9dc6`.
This verifies persisted outcomes and their staged legal metadata; it is not a
fresh gameplay reroll.

The initial verifier incorrectly rejected valid deaths exactly at T=300 and
included outside states in its decisive-only tally. Correcting those predicates
and rerunning the full audit produced PASS without changing the bank. The
overwritten initial FAIL JSON was recovered from the root's captured file-read
output and preserved as `bank-independent-verification-initial-fail.json`
(SHA256 `cd6e31663623b271b23f6ffda56ce2eb57b9f5eb617dc8d106ff7e07c3f7c1af`);
the corrected receipt documents that recovery. Independent review of the frozen
launch slice also found no remaining actionable issues.
