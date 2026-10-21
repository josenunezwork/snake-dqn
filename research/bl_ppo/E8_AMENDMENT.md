# E8 pre-data amendment — 2026-10-21

Status: OWNER_AUTHORIZED_PRE_DATA. E2 has not run. This sibling amendment supersedes
only the clauses named below in INTENT.md; the original intent remains byte-identical.
Authority: owner request 2026-10-21; big-lever round-02 JUDGMENT.md §(f) E8/E9,
round-03 §(f), round-04 §(f) item (11), round-05 §(f), and QUEUE.md.
No Colab calls or dispatch are part of this implementation. Development only;
a funding GO permits replication planning, never serving or release.

1. The critic alone receives `(5000 - frame) / 5000`. The actor receives no clock.
   Gamma is one. Reward is alive logical mass / 1000 each frame. Death and H5000
   bootstrap zero; a collector rollout cut bootstraps the next state's critic value.
2. Before actor unfreeze, refit the critic using at least 2M transitions from the
   frozen stochastic initial actor, with PopArt target normalization and an
   output-preserving rescale. On held-out complete-life, late-body L >= 800 rows,
   require explained variance >= .1 AND MSE below a fitted time/length baseline.
   Hold out whole worlds, fit the baseline on training worlds, retain support
   counts. Missing support or failed qualification yields INCOMPLETE (critic).
   Tiny excluded preflight may bypass the 2M count only while scientific=false;
   it cannot admit E2 or E6 and cannot prove critic preparation.
3. Every 1M transitions, measure KL(B8||policy) and greedy disagreement by length
   decile on a fixed 10k ordinary-state bank, with no new imitation loss. Report
   stochastic versus greedy mass, natural L >= 800 decisions, realised departures
   by length decile, and occupancy per mix by length <400/400–799/800–1199/>=1200
   and frame <2000/2000–3499/>=3500. The 2M natural late-body exposure floor is
   reporting only, separate from the enforced critic-refit transition count.
4. Per 1M, report realised non-greedy runs >=2 and >=3 consecutive L>=800 decisions
   per life (including surviving/censored lives), unique lives with each run,
   denominators, and the mass outcomes of lives containing each run. Carry open
   lives/runs across rollout and reporting boundaries; reset at hero life end.
5. Endpoint reports retain the exact mass identity:
   `delta = [sum_both_alive(Lcandidate-LB8) + sum_candidate_only Lcandidate
   - sum_B8_only LB8] / 5000`. Report by mix and B8 death-frame x last-length
   strata, rescue-to-H5000 and rescue-then-redeath counts, candidate growth slope
   at L>=800, and the unchanged per-mix survival guards. A delayed death is not
   a full-horizon rescue. All ledgers use terminal-hero scoring with dead frames zero.
6. Preserve the matched 100k-window >3x B8 boost scientific tripwire, as explicitly
   stated in round-02 E8(6). Add an endpoint sampled-boost share <.005 contract
   floor FLAG, not a KILL. These are distinct upper and lower checks.
7. GO retains pooled mass point >=+10, scripted paired no-death point delta >=0,
   and survival band v2, and additionally requires pooled LB90>0 and Holm R/F
   superiority at family alpha .05 for E2. E6 has a separate .05 route with Holm
   R/F. KILL/INCOMPLETE precedence remains unchanged. R-F is descriptive.
8. Before scientific collection, freeze E9 operating characteristics with measured
   T23/S1 paired SDs by mix, stratified variance `sum(s_mix^2/n_mix)/9`, shared-row
   correlation sensitivity, mass effects 0/20/40/60/80, and survival deltas
   -.05/0/.03/.06. Require joint E2-or-E6 funding at +40 >=.60 and joint family
   null pass <=.10. Escalate only prospectively to 128/48/48 if needed; failure
   at that size blocks admission rather than silently changing thresholds.
   Round-05 future E16/E17/32M forecasts are conditional simulations, separately
   labelled from the primary E2/E6 rule and from actual observed efficacy.

E4 manifest semantics, E4-dependent 4M/8M/20M sizing, two-arm caps (20 units,
10 L4 allocation-hours, 48 elapsed hours), fixed opponents, natural resets,
endpoint-only one-look inference, immutable exports and no retry are unchanged.
The 2M critic refit is charged inside each arm's whole-session forecast. Old
96/32/32 bank exports remain immutable; any larger allocation needs a fresh audit.
