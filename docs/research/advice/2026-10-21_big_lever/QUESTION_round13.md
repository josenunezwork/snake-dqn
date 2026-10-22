# Round 13 — GPU-trained components that make CPU search cheap (owner question, 2026-10-21 ~21:30 PT)

## Owner's words
"can we use gpus to train/ build things that cpus can benefit from and vice versa?" → Claude proposed learned
components for the served exact-sim search; owner: "ask advisor about this first".

## Goal (unchanged)
A served snake on THIS Mac (CPU + Apple GPU/MPS; no CUDA; serving must stay on the Mac) that clearly beats bare B8 s3
with a claim the owner can trust (Mac strict gate + serving qualification + owner OK). Colab (L4/T4/A100/H100, 3
concurrent assignments, ~1,130 u, budget effectively unlimited) is for training/labels/evaluation only.

## Facts (verify; receipts under big-lever/exp and LIVE_FACTS.md; read round-11 and round-12 JUDGMENTs)
- Only lever that ever clearly beat B8: exact-sim search acting (T3-dense +92 pooled / +122.8 on s3, measured on scripted
  census worlds). Claim A (distil search decisions into B8) KILL; E3 (richer inputs) KILL; Path B deliverability STOP.
- L1′/Y1 (Mac): per 6×H300 batch frozen 10.6–26.3 s, mixed 7.6–11.0 s, scripted 0.6 s; served bar ≈ 1.46 s per root at
  W=4 — measured with the STOCK Python opponent-veto hook; `opponent_v8` = 78–90 % of frozen/mixed batch time; non-veto
  remainder 1.3–2.7 s. Native S2 veto: 1.9× whole-workload in S0; K-1 (native veto in branch rows) action identity PASS
  294/294; K-1 timing Thu 20:00–22:00 (bar ≥ 5× on opponent_v8). Round 12: even with K-1 PASS, arithmetic ~4× short.
- Colab L4 vCPUs 7–10× slower per worker than the Mac for this sim; GPU-resident exact sim (E1b) 0.217 world-steps/s → STOP.
- Opponents are deterministic policies (frozen/scripted sealers, v8-veto vector61); hero = B8 s3 (1.6M-param CNN).
- E4: ~1,440 hero transitions/s per L4 for PPO collection. P1 (100M PPO) in progress.

## Claude's proposal to evaluate (do not take it as given)
GPU-train small networks that replace the expensive parts of the CPU search, keeping the search in charge:
(1) learned opponent model replacing exact opponent simulation inside branch rollouts; (2) learned value cutoff
(short exact rollout + value net instead of H300/H500 tails); (3) a pruning prior (search 2–3 of 6 actions). And the
reverse direction (CPU sims generate data for GPU training) is already the PPO/label pattern. First step proposed: train
a small opponent model on one L4 (<10 u), measure opponent-action agreement and search-decision agreement on the 294 K-1
fixture cases plus Mac timing; kill if < ~95 % search decisions unchanged or < 5× on the opponent part.

## Open question
Is "GPU-trained approximations that make CPU/Mac search cheap" the right lever for reaching a served gain over B8 on this
Mac — which component (if any) should come first, what is the cheapest decisive first measurement, what are the failure
modes (approximation error compounding over long rollouts, distribution shift when the hero's behaviour changes,
Claim A-style failure, identity/trust for a non-exact searcher), and how should it be sequenced with K-1 and P1 this week?
