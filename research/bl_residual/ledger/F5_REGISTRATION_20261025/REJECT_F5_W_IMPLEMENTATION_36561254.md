# REJECT F5-W implementation 36561254

**REJECT this implementation packet.** Astra, independent implementation review. This rejects two operational implementation defects; it does not change the adopted W selection, registered scientific design, or exploratory interpretation. No current draft is accepted for dispatch or fitting. Repair, source-bound regression proof, a new immutable packet, and independent re-review must precede Claude's L4 capture.

Source checkout: `/Users/josenunez/Projects/ml/snake-dqn-f5w`, branch `sol-f5-w-build-20261025`; clean before and after verification. HEAD `2e20aa1c7686bc71920faba63c1c671215c306e8`; implementation `365612540f16b068c7748eb166e150fcb6750a9e`. All source paths below are relative to this checkout. Code was read-only. No commit, edit, dispatch, Colab call, or tmux operation occurred. Only governance/published metadata and synthetic tests were used; selected-training archives, original label payloads, checkpoints, D4096 outputs and `t34-round0-study-01/` were not opened. The named selection receipt was used as governing authority, not as another selection or efficacy look.

## Bound authority and packet

- Adopted registration `F5_REGISTRATION_ADOPTED_20261025.md`: `4e218263f169334415968f02eb2d767e1ff136dfc3ccf470090dd474b31dd6a8`.
- Selection `F5_SELECTION_W_20261025_20261026T032844Z.json`: `b4a880816bcd050cc36aebed7f73547a37a5fc6890889fff7203771a6449d3c3`.
- `F5_W_PREFIT_20261025/final-02/REVIEW_PACKET.json`: `aa991733d75275213e68b27c3a73163fc3be9b6031d0adbf6c7e4d301f3407f4`.
- `REVIEW_BUNDLE.json`: `7f9b5699912457cec9fb9ff3aaefcd647d902bf27172f2821edc04bb76b03c18`.

Independently recomputed all 14 packet file hashes/sizes, all 21 source hashes against both working files and the implementation Git object, and all 40 explicit metadata/governance bindings in authority.json. Prohibited-reference entries were not followed. The selected-960 ruling, common-environment/RNG amendment, batch-order erratum and exploratory amendment control precedence. Unresolved historical overlap remains accepted risk for BOTH contrasts, not verified complete historical freshness; an actual discovered collision invalidates F5.

## Blocking defects

**P1. Cross-job environment identity cannot support the proposed L4 sequence.** `research/bl_residual/f5_job.py:320-326` compares the entire actual capture to the prior sealed capture. Captured fields include physical GPU UUID (`:211`), the UUID-bearing driver query (`:180-188`), CPU affinity (`:220-222`) and installed metadata hashes. The packet proposes separate capture, prepare and fit jobs. Ordinary supervisor completion releases its assignment (`research/colab_cli/supervisor.py:1212-1230`), so the next allocation has no guarantee of that physical UUID/host identity and will refuse. A supported retained L4 route does not exist: supervisor chain explicitly requires H100 (`:1885-1888`), while the mandatory dispatch chain requires A100 (`tools/ops/dispatch.py:538-549`). The generation-specific venv in `f5_colab.sh:15-18` also requires care when comparing installation metadata; path-sensitive RECORD files are not a portable build identity.

Required repair: explicitly separate invariant common-build/scientific identity from per-assignment observed hardware provenance, with independently reviewed allowed comparisons, while keeping C and W in the same verified runtime; OR provide and independently qualify an actual L4 lifecycle that preserves the binding. Do not silently drop hardware/build checks or change the adopted environment. Any amended comparison must be prospectively documented and judged before execution. Test distinct legitimate assignment metadata, true build/precision/source drift, and the actual capture-to-prepare-to-fit handoff. Existing prepare tests mock environment admission and cannot establish this boundary.

**P1. The designated CUDA family canary cannot produce qualifying evidence.** `f5_package.py:444-469` declares a GPU family probe but selects `F5_MODE=prepare`. `f5_job.py:420-448` only opens permitted training metadata and generates a CPU plan; `f5_fit.py:213-236` explicitly uses a CPU generator. There is no CUDA six-action selection on that path. `research/colab_cli/canary.py:377-398` requires an observed source-native CUDA argmax/multinomial over six actions for a CUDA job. Thus even otherwise successful preparation cannot yield accepted family-canary evidence. GPU hardware capture alone does not satisfy that rule.

Required repair: implement a genuine source-native, sealed-disjoint qualification path that supplies the required CUDA action observations and registered timing/parity proof without running either scientific fit. Do not fabricate observer evidence, add an unrelated warmup, or weaken the qualified broker. Keep plan preparation and transport qualification distinct where their contracts differ. Test this mode against the real canary evidence/admission rules. Before main admission, its proof must also cover the complete actual science upload inventory and required producer exports (`canary.py:280-291`); the current prepare-only inventory/exports do not do so.

## Independently verified portions

(a) Read `f5_fit.py:297-325` directly. `known = exact & legal & resolved mask`; pairs require an exact survivor and exact dier. W's additional predicate is ordinary AND pair survivor equals masked B8 argmax. Only those pairs get 4; all other pairs get 1. Division is the original pair count. C uses F4's pair mean. Growth, all-six-action ordinary KL, separate weight/bias L2 and Adam settings remain common. An additional independent scalar enumeration on a mixed 16-pair synthetic batch (four weighted pairs) reproduced C/W values and gradients exactly using the same softplus primitive. This is synthetic algebra evidence, not scientific-input inference.

(b) Independently recounted the lawful status inventory: 960 selected/eligible lives, 372 parents, 86,523 rows = 79,156 doom + 7,367 ordinary. All nine ordinary-only siblings are excluded; no replacement. Every selected life is observed. Original folds recompute nonzero under the frozen fold function; the 167 excluded parent IDs are absent from eligible rows. The first 192 selected source records retain the D256 namespace. No label payload was needed for these checks. Compact-package label fidelity remains supported by the bound packaging/round-trip receipts, not independently decoded here.

Independently checked all 684 allocation parents/seeds are unique and each seed matches the registration's SHA constructor. Groups: archive 64/22/21/21, probe 11/11/10, gameplay 128 per mix, product 128, timing 3 per stream (12 total). The packet allocation also equals the frozen round-0 constructor and adopted digest `1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7`. This verifies the accepted metadata allocation, not omniscient historical collision clearance.

(c,d) Registered defaults are seed 3403, 2,000 updates, batch 256, 128 doom then 128 ordinary; the common materializer creates 512,000 ordered indices with the pinned CPU generator and life-uniform sampler. Exact production plan bytes remain unmaterialized and unaccepted. Full-state continuation records residual, Adam, counters, loss history and typed RNG states; synthetic interrupted/resumed and serialized-state tests pass. CUDA resume equivalence is not established by CPU tests. Independently compared the entire historical fit source AST to current source: the only changed function is non-fit `heldout_metrics`; recipe, cache, losses, sampler and fit remain identical to registered F4. The F5 C trajectory test matches the F4 synthetic trajectory.

(e) Timing specification retains inclusive 150-second transfer and 300-second staging bars, the complete 45,397,643-byte private-inventory ceiling, twelve disjoint timing fixtures, conservative fixed workloads, and base/effective envelopes. The 10/30/200-minute subphase reservations sum to the 240-minute paired-fit envelope. No measured feasibility is established. Drafts correctly retain scientific_admission=false. Create-only claims, hash-bound stage ledgers and manual retry prerequisites pass synthetic tests; downstream pure analyzers are not themselves a demonstrated production one-look runner. Publish selection before work; seal all units and consume the one legitimate stage claim before each registered numerical look. No outcome-dependent partial summaries or repeated looks are admitted.

## Independent verification

Ran with `./venv/bin/python`, nice 15, CPU, OMP/OMP_THREAD_LIMIT/MKL/OpenBLAS/Numba/vecLib threads all 1, bytecode disabled, pytest cache disabled, one process (`-n 0`), fresh test artifacts under the authorized Astra directory:

```text
python -m pytest -q -n 0 -p no:cacheprovider -o addopts='' --basetemp=<authorized-fresh-test-directory>
  tests/test_bl_residual_f5_fit.py tests/test_bl_residual_f5_data.py
  tests/test_bl_residual_f5_rules.py tests/test_bl_residual_f5_timing.py
  tests/test_bl_residual_f5_job.py tests/test_bl_residual_f5_package.py
  tests/test_bl_residual_f5_protocol.py tests/test_bl_residual_fit.py
  tests/test_bl_residual_f4_fit.py tests/test_bl_residual_f4_fit_binding.py
  tests/test_bl_residual_factory_masks.py
```

**130 passed in 9.14 seconds; exit 0.** The macOS harness restarted with a single OpenMP runtime. The first invocation's output session was not retained; the second invocation above is the evidenced result. Neither these passing tests nor Sol's 166-test receipt covers the two operational defects. A separate focused read-only reviewer corroborated the lifecycle/canary findings; the root independently inspected their decisive code references.

## Exact next operational step and gate

**Now: return to Sol for the two bounded repairs and a newly pinned immutable review packet. Claude runs no existing final-02 draft.** No ACCEPT_F5_FIT receipt is issued, and this ruling cannot be serialized as one. No scientific rule, N, loss, arm selection, look count or bar is amended here.

After repair and independent acceptance of the corrected lifecycle/qualification implementation, Claude proceeds in this order through the qualified ops wrapper and fresh manifests/output/state directories:

1. Dispatch the corrected metadata-only L4 capture packet (`research/bl_residual/f5_colab.sh`, `F5_MODE=capture`, `F5_DEVICE=cuda:0`; Python command `-m research.bl_residual.f5_job capture-environment --output "$COLAB_CLI_ROOT/output/environment.json"`). Gate: current pricing/accounting and transport/source pin accepted; complete actual environment receipt exported, hash-verified on Mac, and runtime disposition verified under the repaired lifecycle.
2. Seal a new control using that reviewed environment identity and permitted selected-training package. Dispatch corrected `F5_MODE=prepare` (`-m research.bl_residual.f5_job prepare --inputs "$COLAB_INPUT_DIR" --output "$COLAB_CLI_ROOT/output/prepare"`). Gate: Mac-verify the typed actual 2,000x256 plan, original row identities, initial/final generator states and all source/corpus/environment/allocation bindings; publish the plan seal before either fit. Bind the actual control and initially empty paired ledger. A Mac synthetic plan cannot substitute.
3. Run the repaired source-native timing/parity/family qualification against the twelve timing-only fixtures and complete final inventory. **No executable timing command exists in the current pure f5_timing module; Sol must supply and independently qualify it.** Gate: actual CUDA action/scanner evidence, same-source parity, complete measured stage forecasts, inclusive transfer <=150 seconds and whole staging <=300 seconds, aggregate remaining envelopes, immutable exports, verified runtime disposition and independent actual-plan/operational packet acceptance. Calibration outcomes remain sealed.

Only after these gates may Claude dispatch the independently accepted C-then-W 2,000-update fit packet. Thereafter S2/S3/S4 retain their fixed ordered one-look gates. Any future GO is exploratory advancement only; Mac strict/serving qualification and owner's release authority remain separate.

Created UTC: 2026-10-26T04:22:44.708495+00:00
