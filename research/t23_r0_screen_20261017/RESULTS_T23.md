# T23 results: R0 N=3 static-pool 8M screen - KILL (2026-10-17)

Pre-registration: `PREREGISTRATION_T23.md` (+ amendment 1). Training: Colab A100-SXM4-80GB, commit a049e92,
five seeds 7101-7105, 8M learner transitions each, all clean exits, no tripwire. Evaluation: evaluator 46fe917,
three Colab runtimes (2x A100, 1x L4; CPU-only), each passed the 4-world Mac parity check; 8M only (2M/4M
descriptive were skipped to save time; they do not enter the decision). Full verdict: `results/t23_verdict.md`/`.json`.

**Verdict: KILL.** Mean equal-mix G = -80.8 mass vs B8 (one-sided 90% t limits -96.1 / -65.5), all five seeds
negative (-60 to -111). By mix: scripted -101, frozen -61, mixed -80. Survival roughly unchanged or better
(frozen no-death 0.76 vs 0.68): the fine-tune traded mass for caution.

Implication: the long league run as designed (B8 warm start + R0 + static/PFSP pool) has no learning rationale.

Operational notes: concurrent `mkdir -p` from two Colab runtimes created two Drive folders named `evaluation`
(shard 2 landed in `evaluation (1)`, merged on the Mac); pre-create shared directories before launching workers.
