# Standing A100 Development Sandbox (B1/B2/B3 Probes) and Prospective Child Collector v2

## Question & Goal

Under the owner-adopted Standing A100 Development Sandbox rule (2026-10-26), can residual/auxiliary heads, cosine/gated/hinge loss variants, or direct action-survival veto classifiers with local crop geometry trained on the admitted D1024 counterfactual corpus (`730` lives / `291` parents / `65,825` rows after excluding `248` fold-zero/census/R30-val parents, split 4:1 by parent salt `a100-dev/sandbox-b/split-v1` into `52,704` train / `13,121` val rows with `182` validation rescue rows) teach or assist `b8-s3` to retain `>= 70%` micro rescue agreement while keeping ordinary-state flips `<= 1%` and ordinary harm count `<= 1`—and can host-side sandbox supervision capture exact direct-child `wait4` CPU nanoseconds without modifying the qualified `research/colab_cli` transport?

## Timeline of Versions & Steps

| Date | Step / Version | What Was Run / Built | Verdict / Status |
|---|---|---|---|
| 2026-10-26 | Sandbox admission & input registry (`tools/sandbox/README.md`) | Built `tools/sandbox/{admission,preflight}.py` and `SANDBOX_INPUT_REGISTRY.json` admitting filtered D256 (`136` lives, `12,512` rows) and D1024 (`730` lives, `291` parents, `65,825` rows; `248` excluded parents) under a 10-line `sandbox-manifest.json` contract (`60 min` base / `120 min` effective runaway guard, `16 u` at `8 u/h`). | `ADOPTED` |
| 2026-10-26 | Sandbox B-1 frozen-trunk residual sweep (`research/sandbox_b_heads/README.md`) | Swept 60 configurations (`3` heads: F4 linear, length/margin-gated 2-layer MLP, cosine-normalized residual $\times$ `5` ordinary KL weights `{1,2,4,8,16}` $\times$ `4` survival-pair weights `{1,2,4,8}`, 250 Adam updates each = 15,000 updates) on frozen 256-D B8 features. | `NO_CANDIDATE` |
| 2026-10-26 | Sandbox B-2 bounded successor-head sweep (`B2.md`, `B2_PACKET_02.md`) | Swept 48 configurations (`normed_linear`, `cosine_fastgain`, `supervised_gated_normed` $\times$ `kl_pairs` vs `hinge_top` $\times$ ordinary:survival ratios `{0.5,2,8,32}` $\times$ `500` row-uniform vs `1,500` decisive-life-balanced updates = 48,000 updates, evaluated across 5 switch margins $\tau \in \{0, 0.1, 0.2, 0.4, 0.8\}$ = 240 rows) on A100 (`b2-packet-02`, commit `5525092d`). | `DO_NOT_CROSS` (`0/240` promising) |
| 2026-10-26 | Sandbox B-3 retrieval, geometry & direct survival veto (`B3.md`, `B3_PACKET02_FIXES.md`) | Diagnostic 0 cosine retrieval (`k in {5,15,50}` in `phi` and `[phi, q_base]`) + 12 neural configurations (`phi`, `phi+q/context`, `phi+q/context+12D_geometry` $\times$ `500` / `1,500` updates = 6,000 updates, evaluated across 12 $\tau_{\text{veto}}$ thresholds $\times$ 3 survival advantages = 432 points) after Argon packet-02 neck-TTL and strict-veto fixes (`7f84306e` / `8fb1c87a`). | `DO_NOT_CROSS` (`0/432` promising) |
| 2026-10-27 | Prospective Sandbox Child Collector v2 (`tools/sandbox/SUPERVISION_COLLECTOR_V2.md`) | Built `tools/sandbox/supervisor_v2.py` and `launch_v2.sh` wrapping `subprocess.Popen` to capture direct-child `wait4` user/system CPU nanoseconds and lifecycle journals without editing `research/colab_cli`. | `ACCEPT` (offline verified; `NOT_ADMISSIBLE` for live served panels pending 10-min host canary) |

## Final Verdict & Key Numbers

- **Sandbox B-1 through B-3 Overall Verdict:** **`NO_CANDIDATE` / `DO_NOT_CROSS`** — across all three A100 sandbox generations (60 B-1 configs, 240 B-2 config-threshold rows, and 432 B-3 veto sweep points plus k-NN retrieval), zero points satisfied the promising bar (`micro rescue retention >= 0.70`, `ordinary flip_rate <= 0.01`, `ordinary harm count <= 1`; [`research/sandbox_b_heads/README.md`](../../../research/sandbox_b_heads/README.md), [`research/sandbox_b_heads/B2.md`](../../../research/sandbox_b_heads/B2.md), [`research/sandbox_b_heads/B3.md`](../../../research/sandbox_b_heads/B3.md), [`docs/notes/big-lever-experiments.md`](../../notes/big-lever-experiments.md)).
- **Admitted Sandbox Corpus (`tools/sandbox/SANDBOX_INPUT_REGISTRY.json`):**
  - `d256-allowed.npz`: `136` lives, `12,512` rows (`0` excluded).
  - `d1024-allowed.npz`: `730` lives, `291` admitted parents, `65,825` rows (`52,704` train / `13,121` validation rows; `182` validation rescue rows) after excluding `248` parents (`64` fold-zero + `128` census + `64` R30 validation, with `8` overlap; [`tools/sandbox/README.md`](../../../tools/sandbox/README.md)).
- **B-1, B-2, and B-3 Key Diagnostic Numbers:**
  - **B-1 (`packet-02`, 60 configs, 15,000 updates):** Frozen-trunk residual heads remained locked near bare B8 (`val_action_acc = 0.873180`, `val_rescue_top1_rate = 0.010989 = 2/182`) whenever ordinary KL held ordinary disagreement in check ([`research/sandbox_b_heads/README.md`](../../../research/sandbox_b_heads/README.md), `/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-sandbox-b-20261026/`).
  - **B-2 (`b2-packet-02`, 48 configs / 240 rows, 48,000 updates):** Completed on A100 (`83.14 s` wall, `0.18 u`); fast-gain cosine, nonzero-initialized supervised danger gating, and decisive-life-balanced sampling shifted validation rescue retention by at most `1/182` (`0.0110 -> 0.0165`) at acceptable ordinary disagreement (`0.1257`), with `0/240` points meeting the crossing bar (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-sandbox-b2-20261026/a100-attempt-01`, [`research/sandbox_b_heads/B2.md`](../../../research/sandbox_b_heads/B2.md)).
  - **B-3 (`b3-packet-02`, k-NN retrieval + 12 neural configs / 432 points, 6,000 updates):** Completed on A100 (`83 s` wall, `0.19 u`); bypassing the Q-margin via a direct survival veto and 12-D local crop geometry (`7` crop flood features + `5` v8 scalars) lifted rescue top-1 agreement only when `ordinary_disagreement` spiked far above the `<= 1%` ceiling (up to `0.1868 = 34/182` rescues at `0.2968` ordinary disagreement), leaving `0/432` points promising (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-sandbox-b3-20261026/a100-packet02-attempt-01`, [`research/sandbox_b_heads/B3.md`](../../../research/sandbox_b_heads/B3.md), [`research/sandbox_b_heads/B3_PACKET02_FIXES.md`](../../../research/sandbox_b_heads/B3_PACKET02_FIXES.md)).
- **Prospective Child Collector v2 (`tools/sandbox/SUPERVISION_COLLECTOR_V2.md`):**
  - Passed `621` broad compatibility tests in `727.80 s` (`4` deselected) and `263` focused collector/transport tests; local smoke child (`pid=62259`) recorded **`8,707,000 ns` user + `3,973,000 ns` system = `12,680,000 ns` CPU** over a `17,326,250 ns` span (`26` ordered lifecycle records in `lifecycle.jsonl`, `complete.json` SHA-256 `f66c4490...`; `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/sandbox-collector-v2-20261027/`, [`tools/sandbox/SUPERVISION_COLLECTOR_V2.md`](../../../tools/sandbox/SUPERVISION_COLLECTOR_V2.md)).

## What Was Learned

1. **Single-Frame Representation Overlap Blocks Post-Hoc Residual & Veto Heads:** Across B-1, B-2, and B-3, neither frozen 256-D B8 trunk features (`phi`), `[phi, q_base]` context, nor 12-D local crop geometry separated H300 decisive rescue states from ordinary states cleanly enough to recover `>= 70%` of rescues at `<= 1%` ordinary flip rate. Any threshold or weight aggressive enough to flip rescues triggered massive false-positive overrides on ordinary states.
2. **Local Crop Geometry Cannot Reconstruct Full-Board Tail Reachability:** Even after fixing neck-occupancy TTL blocking (`ceil(length)` for previous head, `ceil(length) + 1` for boosted intermediate cell in `B3_PACKET02_FIXES.md`), the local $31 \times 31$ crop cannot see full-arena multi-snake enclosures or off-crop tail release.
3. **Zero-Touch Subprocess Instrumentation:** Wrapping `subprocess.Popen` in a process-local facade (`tools/sandbox/supervisor_v2.py`) captures exact `os.wait4` `ru_utime`/`ru_stime` nanosecond counters without modifying a single byte of the qualified `research/colab_cli` tree.

## Why It Closed / Current Status

- **Sandbox B-1/B-2/B-3 Head Line:** Closed after B-3 (`DO_NOT_CROSS`), confirming that single-frame post-hoc heads on `ego2s-bv8` cannot replace online simulator rollout search (`exact-stop-l2-prime`).
- **Sandbox Infrastructure (`tools/sandbox/`):** Active and retained in the repository (`admission.py`, `preflight.py`, `supervisor_v2.py`).

## Kept Files in This Lineage

- [`tools/sandbox/README.md`](../../../tools/sandbox/README.md) — Primary documentation for the Standing A100 Development Sandbox admission, input registry, and preflight workflow.
- [`tools/sandbox/SUPERVISION_COLLECTOR_V2.md`](../../../tools/sandbox/SUPERVISION_COLLECTOR_V2.md) — Technical specification and Argon review record for `tools/sandbox/supervisor_v2.py` and `launch_v2.sh`.
- [`research/sandbox_b_heads/README.md`](../../../research/sandbox_b_heads/README.md) — Sandbox B-1 60-configuration frozen-trunk residual sweep specification and resume protocol.
- [`research/sandbox_b_heads/B2.md`](../../../research/sandbox_b_heads/B2.md) — Sandbox B-2 48-configuration normalized/cosine-fastgain/gated-residual sweep specification.
- [`research/sandbox_b_heads/B2_PACKET_02.md`](../../../research/sandbox_b_heads/B2_PACKET_02.md) — Sandbox B-2 Packet 02 handoff (`5525092d`).
- [`research/sandbox_b_heads/B3.md`](../../../research/sandbox_b_heads/B3.md) — Sandbox B-3 retrieval, 12-D crop geometry, and direct survival veto specification.
- [`research/sandbox_b_heads/B3_PACKET02_FIXES.md`](../../../research/sandbox_b_heads/B3_PACKET02_FIXES.md) — Argon review fixes for B-3 Packet 02 (neck TTL blocking, zero-vote sentinel `1.0`, and strict survival inequality).
