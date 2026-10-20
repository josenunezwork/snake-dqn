# T34 Round 0 (teacher premise test on Colab) — pre-registration package, Sol spec (2026-10-20 ~12:30 PT)

Owner (2026-10-20): "im fine with using colab credits ... if they speed up our efforts and research lanes"; owner
delegated next-step calls to Claude. Round 0 = FINAL_PLAN §8 premise test, now with the Amendment 2 teacher
(DESIGN_V2_SHADOW.md Amendment 2, P7_STATISTICS.md, SUCCESS_CONTRACT.md on main): B8 s3 + teacher acting
(beta=1, bisection on shadow doom, horizon rescues executed) vs bare B8 s3, paired worlds, NO deadlines (offline
teacher, labels not produced for fitting). Purpose: does the deterministic search teacher, acting, beat bare B8
by enough to justify label rounds (Claim A lane)?

1. research/t34_round0/PREREGISTRATION_ROUND0.md (DRAFT_PENDING_OPERATOR_RATIFICATION): question, comparator,
   arms, 240 paired H5000 worlds (96 scripted / 72 frozen / 72 mixed), fresh disjoint namespaces per mix
   (collision audit vs research/, docs/, scripts/, all prior banks incl. research/t34_pathb retention banks),
   paired-world inference per P7 (fixed controller), GO = scripted no-death >= 0.66 AND >= +0.10 vs paired bare
   AND pooled mass >= +40 with LB90 > 0 AND boost share in [0.5%, 10%] (FINAL_PLAN §8, comparator B8 s3; survival
   guards A/B per P7); STOP/INCONCLUSIVE rules; Phase 0 is a separate cross-check only. Platform: Colab L4 lanes
   (development evidence; Mac strict gate unaffected); both arms of a pair run on the SAME runtime (no cross-
   platform pairing); per-world hashes + A2-style replay of a 10% sample on the Mac (report divergences).
2. Colab entrypoint + manifests (research/colab_cli conventions, public inputs release s1-inputs-20261019,
   colab_cli qualification check), chunked so est <= 180 min, timeout 360, export per chunk, <=3 lanes; chunk
   size from the A5 measurement the operator supplies (leave a parameter); unit cap: compute it (priced 2.0 u/h)
   and state it; 20% reserve.
3. Mac preflight (2 worlds, 500 frames) of the real entrypoint; validation script; operating-characteristic note.
Rules: Mac only; no live Colab/tmux; <=2 CPU threads (Phase 0 running); do not run any arm; commit/push often;
merge to main; README with exact ratify-and-dispatch commands.
