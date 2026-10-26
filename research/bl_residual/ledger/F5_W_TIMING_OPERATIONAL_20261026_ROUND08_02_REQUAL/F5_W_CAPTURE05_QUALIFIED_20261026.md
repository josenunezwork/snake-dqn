# F5_W_CAPTURE05_QUALIFIED_20261026

2026-10-26T12:07:33.344176+00:00 · Astra independent F5 review · read-only repository and saved evidence.

**QUALIFIED: capture attempt 05 passed v5 environment capture and disjoint source-native CUDA transport on the requalified CLI tree.** This is not full F5 timing qualification, a scientific fit/result, final science inventory qualification, a serving qualification or release authority.

The saved job is `bl-f5-w-capture-requal-3b32d755-05`, generation `cb68d49f5ebb4b189836bd7b97b278de`, runtime source `e25f889b742764d29313340eecb41703e323d0dc`. Its CLI tree and the reviewed clean broker's tree both equal `6e8c1982b5c9ed115bfebd3145ad0720b15522fa`. Source publication and execution are distinct: output evidence was published in `75b1442768f37ce83188a2990c9ef0026ee806b5` at 11:43:19 UTC; the post-publication report was created at 11:44:12.432355 UTC and was absent from that earlier publication commit. The independent capture lane verified all 229 published output Git blobs and lossless reconstruction where required.

| Requirement | Independently checked evidence | Ruling |
|---|---|---|
| Frozen v5 environment | Raw successful receipt, adopted schema/build/wheel/control fields; offline `validate_environment` and `compare_environment` pass | PASS |
| Twelve parity workloads | Exactly frozen fixture IDs and seeds 672–683, 12/12 PASS; producer requires allclose with atol=rtol=0.0002 and exact actions | PASS |
| Real source-native CUDA | `f5_qualification.py:101–137` constructs GridBatchSim/ego2s observations and runs B8 fp32 CUDA forward; observer sees package-attributable `Tensor.argmax`, cuda:0, [1,6]. Observer records only successful CUDA-synchronized calls; recomputed canary evidence equals receipt | PASS |
| Transfer/staging | 57.352172708837315 s <=150; 76.89912299998105 s <=300 | PASS |
| Immutable exports | All 10 required exports recovered with exact hash/size/SHA sidecar/completion verification | PASS |
| Terminal/release | PASS, no terminal error/cleanup error, stop_verified=true; saved stop then sessions{} then zero-assignment/zero-rate usage observations | PASS as of saved terminal state |

The latency summary requires an explicit correction to the shorthand prompt: **four** approximately 120-second renewal exec calls failed, each followed by completed affirmative recovery-idle evidence (maximum 8.396584625006653 s). One background heartbeat failed after 37.72717287496198 s. The maximum entire renewal control operation was 262.8373576669255 s, including bounded attempts/recovery; it is not the staging duration or a single CLI call. These nonterminal stalls did not prevent this recorded completion. They do not prove a causal repair effect, future availability, future reliability or a retry entitlement.

The raw reference is `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_W_CAPTURE_AFTER_TRANSPORT_REQUAL_05/output-05/exports/environment.json`, SHA256 `2b5e0713cc907a08688e1d27bcd171669f345a2f25478a4e6e07be9ce2a29e6d`. A future assignment must independently capture and pass the frozen cross-assignment comparison; distinct L4 assignments are not presumed identical. The transport-only receipt explicitly records no scientific optimizer execution and `NOT_MEASURED_TRANSPORT_ONLY` Mac timing. Producer-named paired placeholder exports do not become scientific fits.

Evidence bindings:

- POST_CAPTURE_VERIFICATION_05.json: `8a06712f6700420562016e048f7983b0c633eb7d5cbc291b79e91c65d0fcfa1b`.
- Terminal receipt: `44d8a673490d627d5cc5ba24b697fb1e2e99d344678d865a5e85a7881c32e78e`.
- CUDA qualification: `ccfca738baff0319d640d1e3fef7a7b0b0f8bb272f4d05cb7afb4b1389096c14`.
- OUTPUT_PUBLICATION_05.json: `d2118b68e3e2defab59ebfa3bacce5d2e4fdbd73352b3e06ea926e33eca8d072`.
- Root capture audit ASTRA_F5_CAPTURE05_AUDIT_20261026.json: `3774e0bdf26795d776150ed6892e2e51cf3b8e2bf31ba1e4fffbf96742f54acc`.
- Offline consumer/history supplement ASTRA_F5_REQUAL_SUPPLEMENTAL_20261026.json: `45559d2c7393e485e44b9ecf11230d2c26e790f84f2f792238013657f52ce7f4`. Both qualified consumer checks passed without provider calls; the registry binds acceptance `8356217255f0e4e8f40e27c2fbfcd4785b991d9e594ffc4eed83419063d8847f` and the required synthetic inventory, with no transfer exception.

No repository edit, commit, dispatch, Colab/provider call or tmux operation occurred. No label/outcome payload, D4096 output or t34-round0-study-01 content was opened. Only create-only review files and SHA sidecars were written under the authorized Astra artifact root. Release verification here is saved evidence; no current account query was performed.
