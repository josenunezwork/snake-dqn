# E1 prospective development rule (Round 1 judgment §f)

Frozen before E1 tests, preflight, or GPU data. Development experiment only,
excluded from every registered verdict. This E1 revision retains the original
spike cost/agreement rule; it does not amend any registered study.

CONTINUE (recommend a full port) only if **both**:
- end-to-end GPU cost per dense-search decision ≤ **1/10** of the L4 CPU baseline on the long-body mixes, **and**
- first-action agreement ≥ **99 %** of searched decisions on the S0 fixtures, with 0 unexplained (bug) disagreements.
Otherwise STOP: write `RESULT.md` with the numbers and the reason; no second attempt without a new brief.

Additional E1 engine admission rule: **<20,000 full world-steps/s on L4 => STOP
engine thesis this week**, even if the cost/agreement criteria pass. Apply this
floor to each of the three long-body S0 mixes with the real five opponents.
A full world-step advances all six snakes, including real policy/observation/v8
work and simulator events. Report six branch worlds × rollout depth / synchronized
rollout wall seconds, per case and pooled, excluding explicit warmup. This counts
scheduled world-steps (including dead/illegal branch lanes), never snake-row-steps,
hero transitions, isolated simulator steps, or a policy-only microbenchmark.
Also report the pipeline rate including setup/warmup/readback/CPU label replay;
bootstrap/export and measured allocation cost remain supervisor-receipt overhead.
Tiny CPU preflight rates cannot decide the L4 throughput rule. Missing long-body
mixes, an incomplete port, or a timeout cannot qualify CONTINUE.

Decisive-root audit (CPU-oracle definition, fixed before data): a root is decisive
when the CPU exhaustive search's best legal branch has strictly more alive steps
than the CPU bare-greedy branch. Report first-action agreement, sample count,
classification counts, and branch mass error separately for decisive and other
roots. Do not select this stratum using candidate actions or candidate outcomes.
A zero decisive-root count is explicitly NO_COVERAGE, never evidence of decisive
agreement. This audit adds no replacement threshold to the frozen ≥99 % overall
rule, and gives no release/serving authority. Every disagreement still requires
positive explanatory evidence or counts as an unexplained bug.

Budget: one L4 spike ≤15 units including reserve; an optional separate L4 probe
≤5 units including reserve. First wall/unit cap wins, no automatic retries;
48-hour development window. Claude dispatches, Sol only prepares/validates.
