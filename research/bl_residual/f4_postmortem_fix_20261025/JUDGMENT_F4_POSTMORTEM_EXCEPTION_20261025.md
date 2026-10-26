# Binding pre-data ruling: one F4 harm post-mortem export

2026-10-25 · Astra, independent T34 judge · RATIFIED by this ruling under the owner's explicit request.

**ALLOW ONE DIAGNOSTIC EXPORT; STOP_FUNDING REMAINS FINAL.** This is a narrow prospective exception to the prohibition on recomputation in JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md, solely to recover the missing harm diagnostic and inform the choice of F5's harm-targeted change. It is prospective to this additional diagnostic exposure, not a claim that the underlying archive or original R-A outcome is unseen. It does not reopen F4, amend its thresholds, or grant new efficacy evidence. The successor-plan recommendation was supplied by the owner in this request; the executable scope is fixed here.

The judge read source and published metadata only: no checkpoint, NPZ, raw outcome payload, or D4096 payload was opened or hashed. Repository snapshot: main at 3a830e711c1851c1258912408d520ce1faf248ce; existing untracked exp/mac-lease.json.live and exp/mac-lease.json.lock were left untouched. No code execution, dispatch, tmux operation, commit, or repository edit occurred. These ruling bytes and their SHA sidecar are the only authorized writes by this judge.

## 1. Exact permitted inputs

Repository-relative ledger paths below are rooted at /Users/josenunez/Projects/ml/snake-dqn/. SHA256 values are file-byte hashes. Their sidecars were checked and their bytes matched cached origin/main; no fresh remote-publication claim is made. The operator must establish current publication before payload access.

| Ledger or ruling | SHA256 |
|---|---|
| research/bl_residual/ledger/F4_R_A_20261025/command-packet.json | `443ad83fb6b3cb683761ccc0223d86414a03d08fc6902f8d10e86bf20ea9dd8c` |
| research/bl_residual/ledger/F4_R_A_20261025/funding-look.json | `4b8897198c3ae086689bd1d13afd9a0c5a2ef6af2a66cb00fa31e9e1b869c8fe` |
| research/bl_residual/ledger/F4_R_A_20261025/d4096-not-run-by-rule.json | `f4d535e29c96a5abc4756ed1c8cf56a9e26afcb957e520378cc93aec3c39fd69` |
| research/bl_residual/ledger/F4_R_A_20261025/JUDGMENT_RA_STOP_FUNDING_f9c08082_20261025.md | `be1516a98ac7851ad30cc9694aa1a70b57cd30950ebafe58c4ffbc02ac3deb05` |
| research/bl_residual/ledger/F4_D256_METADATA_FIT_BINDING_20261024.json | `65731d8bf2a4d63e81b18ad5024359604d38fa9e36253ae97bef1f94913d9cba` |
| research/bl_residual/ledger/F4_D1024_FIT_BINDING_20261024.json | `b44ba54e84b68add3f583d66556d37d5eea19aeccad89dcd764e20076dd9bed2` |
| research/bl_residual/ledger/F4_D256_FIT_PHASE_20261023.json | `82b53a3b8ccd4dfd49bac71e77d80e9926946a6225855553390ec4db21686abc` |
| research/bl_residual/ledger/F4_D1024_FIT_PHASE_20261024.json | `1de137562ac5f99e7224ac965f72dc2c9f2312b2c333c7ef3057703c9ffc4e0d` |
| research/bl_residual/ledger/F4_R_A_ARCHIVE_CENSUS_20261024.json | `44577f8a003314e97bcffe29dbb4f44cf51be67ec36f90e972b499c3f07530f0` |

Only the following packet-bound inputs may be opened by the implementation. Repeated table paths designate the same immutable file, not separate archives. Verify SHA/size/sidecar/completion before deserialization, and reconcile checkpoint identities with the fit-phase ledgers. These are metadata-derived identities, not payload verification by this judge.

| Input | Absolute path | SHA256 |
|---|---|---|
| --approval | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d256-metadata-pack-02/inputs/judgment.json | `15b6c842e73f1030b1fd04210e5413469acb8af7850a222b8e412608a9b80111` |
| --archive-receipt | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-archive-census-01/census.json | `44577f8a003314e97bcffe29dbb4f44cf51be67ec36f90e972b499c3f07530f0` |
| --b8 | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d256-metadata-pack-02/inputs/b8.pth | `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8` |
| --d1024-allocation | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d1024-pack-04/inputs/life-allocation.json | `31fe074d72c3fe7eaeb1d30c59622a6f5b4add4d3c6c2b8320048265d5986c4d` |
| --d1024-checkpoint | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d1024-pack-04/d1024-output-l4-04/exports/fit/fit-0001.pth | `da1d265608d7dbb2a68b285c3394cd080cc2653b4e767139dac918076a265093` |
| --d1024-control | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d1024-pack-04/inputs/control.json | `d55e4665f11d7934b6ea5f617129048ed7fda64ca9b307ca9b4cbfb1a323f6fe` |
| --d1024-table | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-archive-census-01/archive-table.npz | `34ad5395be877c7b9fe46b22c5fbac91f6656fe5edbd35967c7e141b77897b4d` |
| --d256-allocation | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d256-metadata-pack-02/inputs/life-allocation.json | `bf19f404e4f707de2ff3d50f1bcc3db1003446a624864504ba1865e26e474bdb` |
| --d256-checkpoint | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d256-metadata-pack-02/d256-output-l4-01/exports/fit/fit-0000.pth | `103d02e9953a45e4c0ea3a7413dac0f56d96d99a0a18c27aeff01a4e90cf0360` |
| --d256-control | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d256-metadata-pack-02/inputs/control.json | `2744be3858b12c9fa55578caf5ab1dece632ec313e2b66f0bbff91e53dbc954a` |
| --d256-table | /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-archive-census-01/archive-table.npz | `34ad5395be877c7b9fe46b22c5fbac91f6656fe5edbd35967c7e141b77897b4d` |

Additionally, the original immutable joint claim may be read for provenance and existing heldout guard validation only:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-claims/F4_R_A_845eaa5b4ad6e68971d5bff0591d668ad9a37a4ebd8231df517a7fcc24d2d273.lock.json`, SHA256 `3d5a9ab63d14104e369ccf443998009d4e85aaa8d38722c87883a770a746da61` (identity from the published STOP_FUNDING ruling). Never rewrite it or manufacture a replacement R-A claim. Metadata receipts explicitly enumerated in the two fit-binding ledgers may be verified as provenance, never used to expand the payload allowlist. Source modules, this ruling, the prepublished implementation manifest and synthetic fixtures are also permitted. No raw union, new checkpoint, additional archive, D4096 payload, new labels, or gameplay is authorized.

## 2. Frozen computation; observation only

Reference implementation pin: `f9c08082b4e298a3d6666e7820e49c7973a16b99`. Current inspected source matches that pin for these relevant files:

| Source | SHA256 |
|---|---|
| research/bl_residual/funding.py | `776b5bed1ac8a573a729bcacd193708d107c3c9f8eedf4236ae921921f94f0eb` |
| research/bl_residual/heldout.py | `c015b74f299464279a89b963c749de81d132445a50b614e1a0e338465aec59e0` |
| research/bl_residual/fit.py | `85c7e77d34e14a3b5036984cc1af92df4e3724f4c9fe453d813a56c1963822a7` |
| research/bl_residual/data.py | `7273c09302bce5c2e09bb8abe92424e70df2b142d83c0a33ec42c36b8026a980` |
| research/bl_residual/model.py | `e00d30e0fd2e752bcd634676a364e0750f316c9ab3fb47251edcd50d87f66c04` |
| research/bl_residual/protocol.py | `8052b6390f61169ff8f4dc4bca0ce116b5c1fcd64bdd0fbb676f4a4658d7dcd6` |
| research/bl_residual/F4_REGISTRATION.md | `66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65` |

Sol shall add a separately named post-mortem entrypoint reusing funding.py's two-candidate heldout call sequence, heldout.evaluate's immutable-candidate/census/ancestry validation and rowwise CPU forward, and fit.heldout_metrics' existing served-argmax, action masks, harmful_exact, harmful_unknown, natural_stream and margin definitions. A minimal optional diagnostics observer may count those existing masks before they are discarded; do not independently reimplement their semantics. Preserve the default call behavior and original once-only guards. Passing the original verified joint claim is allowed only under this new entrypoint's separate create-only post-mortem claim; its existence alone is not authorization.

Do NOT invoke funding.main or funding.evaluate, bootstrap retention, calculate a new funding verdict, alter weights/actions, fit a model, change inference batch size or precision, or change labels, masks, thresholds or archive membership. Other metrics computed internally by the existing heldout path must remain unexported and unread. No new inferential statistics, confidence intervals, significance tests, alternative harm definitions or margin searches are allowed. Simple counters and grouping of the existing masks are the authorized diagnostic instrumentation.

For each dose independently, over natural_stream rows only:
- C = count(harmful_exact): changed action, selected exact/legal death and B8 exact survival, exactly as implemented.
- U = count(harmful_unknown): changed action and selected or B8 outcome not EXACT, exactly as implemented; UNKNOWN is not a confirmed death.
- H = count(harmful_exact OR harmful_unknown) = C + U. Preserve the original conservative H even if U dominates.
- Decompose U into selected-only nonexact, B8-only nonexact, and both nonexact (three disjoint counts). Do not use unknown_selected_coverage as a substitute for U.
- Margin = abs(candidate_Q[selected] - candidate_Q[B8_action]), with the existing bins [0,.01), [.01,.1), [.1,.5), [.5,infinity). No alternative margin or thresholds.

## 3. Exact export schema and exposure boundary

One create-only bundle at `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/postmortem-harm-20261025-01/`: pre-data `plan.json`, once-only `claim.json`, then `diagnostic.json` and `development-parents.json`; SHA sidecars and completion markers (written last) for each. A terminal operational failure receipt is allowed instead of a completed diagnostic. No overwrites, replacement directory or second attempt under this ruling.

`diagnostic.json` schema = `bl-residual-f4-harm-postmortem/v1`, containing only:
- `ruling_sha256`, `implementation_commit`, `source_sha256`, `inputs` (exact path/size/hash), `claim_sha256`, `started_at`, `finished_at`, `platform`, `nice`, `threads`, `wall_seconds`, `status` (COMPLETE, INVALID, or INCOMPLETE), `purpose="DEVELOPMENT_ONLY"`, `f4_disposition="STOP_FUNDING"`.
- `census`: total_rows=5936, natural_rows=560, eligible_lives=71, reserve_lives=64, parents=29, copied/reconciled against the pinned census. Mismatch is INVALID, never an excuse to shrink the set.
- `doses`: exactly two objects, dose 256 and 1024 with checkpoint SHA; each has `totals`, four `margin_bins`, and all 29 `parents` sorted lexicographically, including zeros. A parent entry contains parent_world, totals and the same four margin_bins; bin entries contain lower, upper (null for infinity), and counts.
- Every counts object has nonnegative integers `N_natural`, `N_changed`, `N_changed_both_exact`, `N_changed_B8_exact_survives_selected_exact`, `C_confirmed`, `U_selected_only`, `U_B8_only`, `U_both`, `U_unknown`, `H`. These explicit denominators allow H/N_natural, H/N_changed, and confirmed-death share among changed B8-surviving/both-exact rows without conflating populations. Zero denominators mean undefined, not zero risk.
- Parent concentration is the full per-parent H, C and U count table; no selected top-parent subset or new concentration statistic. Its shares are descriptive count/total arithmetic, with null for a zero total. Margin output is counts only, not individual margins or retrospective threshold optimization.
- `checks`: bins and parents sum to totals; C+U=H; the three unknown components sum to U; natural-row denominator is 560 for each dose; exported H equals the existing heldout_metrics H from that same inference pass. A disagreement with the retained D1024 harm-failure boolean is reported as INVALID diagnostic, preserving both records and STOP_FUNDING; it never rescues F4 or authorizes a rerun.

`development-parents.json` schema = `bl-residual-development-parents/v1`, with ruling/table/census SHAs, `designation="DEVELOPMENT_FOREVER"`, the complete ordered parent_worlds below, and exclusion scope `all successor validation, admission, funding, model selection holdouts, strict gates and serving qualification evidence`. Export parent identities and counts only: no row/life-level outcome records, row IDs, tensors, Q arrays, individual actions, snapshots, trajectories or additional heldout metrics. Full per-parent counts are expressly allowed; raw per-row export is not.

## 4. Permanent parent exclusion

Effective upon adoption of this exception, burn ALL 29 census parents, not merely harmed parents, both models' changed-action parents, or the 64 reserve lives. Every life, row, continuation, clone, snapshot and descendant sharing any listed parent is development data forever, including unexposed siblings. Renaming a namespace, resplitting rows/lives or allocating another index does not restore independence. Future validation collision audits must resolve original ancestry/world identity and exclude these parents; unresolved ancestry fails closed. They may inform F5 development, but no successor may count them as validation evidence.

- `d256-mac-gate-frozen-008`
- `d256-mac-gate-frozen-026`
- `d256-mac-gate-frozen-034`
- `d256-mac-gate-frozen-046`
- `d256-mac-gate-frozen-057`
- `d256-mac-gate-frozen-078`
- `d256-mac-gate-frozen-140`
- `d256-mac-gate-frozen-154`
- `d256-mac-gate-frozen-173`
- `d256-mac-gate-frozen-181`
- `d256-mac-gate-frozen-187`
- `d256-mac-gate-frozen-190`
- `d256-mac-gate-frozen-191`
- `d256-mac-gate-frozen-208`
- `d256-mac-gate-frozen-220`
- `d256-mac-gate-frozen-225`
- `d256-mac-watch6-002`
- `d256-mac-watch6-009`
- `d256-mac-watch6-015`
- `d256-mac-watch6-018`
- `d256-mac-watch6-035`
- `d256-mac-watch6-039`
- `d256-mac-watch6-045`
- `d256-mac-watch6-048`
- `d256-mac-watch6-059`
- `d256-mac-watch6-061`
- `d256-mac-watch6-088`
- `d256-mac-watch6-090`
- `d256-mac-watch6-095`

## 5. Publish-before-read, execution and one-run limit

Before ANY new real payload read or inference, Sol/operator must publish this exact ruling and SHA, the source-pinned implementation and synthetic-test evidence, the exact input/output/schema manifest, and the above development exclusion ledger in the repository; verify local bytes against freshly fetched origin/main and retain the publication commit. Implementation tests may use synthetic fixtures only. Freeze all identities and the separate claim path before reading. This judge's artifact creation is issuance, not Git publication or execution.

Acquire the shared research Mac lease exclusively; wait for other research work to release it, never preempt an unrelated job. Execute on the Mac CPU, nice 15, one process and one math thread, with SNAKE_DQN_DEVICE=cpu and OMP_NUM_THREADS, OMP_THREAD_LIMIT, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, NUMBA_NUM_THREADS and VECLIB_MAXIMUM_THREADS all 1. Require AC, lid open, thermal guard and wake lock. Claim the attempt atomically before payload access; one forward pass per candidate over the pinned table, in a single joint run. Claim remains spent on failure or partial completion. No automatic or manual retry is authorized by this ruling.

Prospective safety envelope: base 30 minutes, effective 60 minutes wall time (one application of the owner's 2x rule), from claim through terminal export; this is a runaway guard, not a new measured forecast. No Colab/RunPod, collection, fitting or gameplay. Preserve failures without selective output. Only hash/count/progress telemetry may be read during execution. After completion, publish the result binding and development ledger before humans/agents inspect the diagnostic outcomes or use them to choose F5. Audits thereafter may inspect the immutable aggregate output; no additional model/table recomputation is authorized.

## 6. Permitted use and Sol task

The completed descriptive result may choose F5's harm-targeted mechanism/change and inform a separately judged, prospective successor registration. It cannot rescue, relabel, refund, extend or rerun F4; permit D4096 access, fitting, R-B or gameplay; claim improved survival, population safety, efficacy or promotion; or change B8 serving. Exposed archive parents cannot select or validate a successor checkpoint, tune a threshold and then masquerade as a holdout, or support fresh-bank claims. Choosing a development design from these diagnostics is allowed; confirming it requires independent unexposed parents.

Sol implementation task: build the narrowly gated diagnostic entrypoint and optional counters described above, preserving the pinned funding/heldout harm semantics and default behavior; prove on synthetic fixtures that confirmed/unknown partitions, unknown-side counts, unchanged actions, legality/mask and tie behavior, bin edges, zero denominators, per-parent sums, provenance failures and create-only refusal work; freeze and publish the implementation, this ruling, exact manifest and permanent 29-parent exclusion ledger before any payload access; hand the operator one exclusive-Mac nice-15 command for the single bounded run, then verify and publish its immutable aggregate/count-only output without reopening funding or running F5.

Memory was used only to locate the earlier R-A context; all binding identities and computation definitions here came from current repository metadata and source. No diagnostic result is asserted.
