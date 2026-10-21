# E13 parity, transfer, and multiplicity rule for E16

Status: prospective development rule. Freeze this file with the E16 registration before
any E16 training, evaluation, or transfer-audit data. It authorizes neither a Colab
dispatch nor a strict or serving decision.

## Purpose and location

E16 compares executable policies on fresh natural-start, whole-life H5000 worlds. A
successful upload, a replay that merely reuses recorded actions, or a weights-only
recovery is not evidence that a Colab-trained policy will behave the same way in that
comparison. This rule separates the evidence required to trust an artifact from the
later evidence required for strict qualification and release.

Round 4 §f, E13 named `research/bl_ppo/PARITY_RULE.md` as the intended location.
The E16 request prohibits editing `research/bl_ppo`; this E16-local file is therefore
the canonical E13 rule for this package. It preserves the adopted E13 requirements
without changing the protected PPO package. A later move must preserve these bytes or
be recorded as a pre-data sibling amendment; it must not silently replace this rule.

## Evidence doctrine

The qualification has three distinct layers. Passing one layer does not pass a later
layer.

| Layer | Question answered | Required evidence | It does not establish |
| --- | --- | --- | --- |
| Engine fidelity | Given the same events, does the candidate engine preserve evaluation semantics? | Receipt 1, including byte-level observations and state/event comparisons | That a learned actor behaves correctly in closed loop |
| Artifact admission | Is this exported actor reproducible and predictively transferable to the Mac development evaluation? | Receipts 2–4, including a fresh closed-loop Mac audit | Strict qualification, product bars, or release |
| Strict/release | Does the admitted artifact meet the active campaign's Mac strict and serving requirements? | The separately registered Mac strict gate and serving qualification | A new parity or platform-transfer waiver |

Training may use independent RNG streams, batching, and declared numerical methods.
Those differences do not permit a change to mechanics, observation availability, or
evaluation semantics. Cross-platform training trajectories are not required to be
identical. Closed-loop evaluation behavior and the evidence below are required.

## Frozen identities and receipt envelope

Before running any receipt, bind the following in a create-only receipt plan:

- the source commit and SHA-256 of every evaluated source/input file;
- candidate, parent, B8, opponent, configuration, roster, seed/root-bank, and
  action-stream identities, each by SHA-256 where it is a file;
- command, platform/CPU architecture, Python and dependency versions, thread/device
  settings, numerical-method declaration, and the rule-file SHA-256; and
- the receipt's exact comparison population, selection rule, expected counts, and
  output paths.

Each produced machine-readable receipt and every exported artifact must be
hash-addressed. Export acceptance requires the artifact and both its `.complete` and
`.sha256` sidecars; a partial export, a checksum mismatch, or a later mutation is not
an accepted export. A receipt must record whether it was produced on Mac or Colab and
must never relabel a same-host result as cross-platform identity.

## The four required receipts

### 1. Engine identity on injected candidates

Drive independent reference and candidate engines with the same pre-recorded actions
and random candidates. The comparison must cover the natural and adversarial cases
needed by the E16 mechanics, including reacting-opponent contexts. It must compare:

- bodies and alive state;
- food ordering, caps, and release timing;
- collisions, rewards, terminal flags, and masks; and
- unchanged observation bytes.

The receipt identifies the first mismatch by root and frame and preserves the inputs
needed to reproduce it. Tolerant numerical comparison is not an identity receipt.
An unexplained semantic mismatch fails this receipt. A declared numerical difference
is admissible only when it has no changed engine semantics and the other required
receipts still pass.

### 2. Policy agreement on ordinary and decisive roots

Run the frozen exported actor and the reference actor in closed loop on both ordinary
and decisive roots. For each policy decision, compare logits, chosen action, and all
policy carry used by the actor. Recompute reacting opponents during this comparison;
recorded-action rescoring alone is insufficient.

The receipt must bind the ordinary and decisive root selectors before data. A selector
may not use the candidate's observed outcome. It must name the root universe, the
reference condition defining a decisive root, its seed list, and its selected-root
digest. If no such frozen selector or no decisive coverage exists, the decisive-root
component is incomplete rather than being replaced by a convenient ordinary sample.

For every disagreement, preserve the first divergent root, frame, state/input digest,
reference and candidate logits/actions/carry, and the continuation needed to explain
the divergence. The explanation must distinguish an allowed declared numerical
condition from a semantic mismatch. A global agreement percentage, including 99%,
does not substitute for this first-divergence audit.

### 3. Durability: exact on-platform resume and real-size export

On each claimed execution platform, compare an uninterrupted run with a checkpointed
and resumed run under the same frozen inputs. The resumed run must restore, or
explicitly drain before checkpointing, all state that can affect future work:

- actor/model and optimizer state, including PopArt where present;
- schedules, counters, and all RNG states;
- worlds, opponent carry, and recurrent state; and
- pending rollouts.

The receipt proves exact equality of the subsequent on-platform execution, not just
equal final weights. It records the compared state/output digests, restart boundary,
and counter reconciliation. It also verifies a real-size export through the
`.complete` and `.sha256` acceptance rule above. The existing small-export transport
qualification is not a substitute for this receipt. A weights-only recovery may be
reported as recovery, but must never be called an exact resume.

### 4. Artifact: hash-bound actor and fresh closed-loop transfer audit

Evaluate the hash-bound exported actor and B8 in fresh, collision-audited,
natural-start closed-loop development worlds on both Colab and the Mac. The actor,
B8, roster, roots, and evaluator settings are those bound in the receipt; no
intermediate or replacement checkpoint is substituted after seeing results.

Let `Delta_platform` be the paired candidate-minus-B8 evaluation delta on that
platform, using the frozen E16 evaluation metric and equal-mix pooling. Let
`B = Delta_Mac - Delta_Colab`. Before treating a Colab efficacy result as predictive
of the Mac strict print, the frozen 90% interval for `B` must lie wholly within both
limits:

- pooled `B` within `[-10, +10]`; and
- `B` within `[-20, +20]` for every mix.

The transfer-audit plan must freeze its fresh bank, world count, interval procedure,
and platform pairing before data. It must report decisive-root divergences from
Receipt 2 alongside the aggregate result. Passing this receipt admits an artifact to
development interpretation only. Mac strict qualification, Watch product bars,
serving qualification, and owner release remain separate requirements.

## Family error and looks

E16 has one funding/ranking decision, not independent OR-routes from E2, E3, E6, or
E10. Those studies nominate an available candidate; their result does not by itself
admit a 100M treatment. Unavailable arms remain `UNRANKED` and are never assigned a
zero result.

Use a one-sided family alpha of **0.10** across every statistical claim used for E16
admission or ranking, including all admitted treatments and all B8 and parent
contrasts (`G` and `J`) that can affect the decision. Produce simultaneous,
world-clustered bounds for that family with the multiplicity procedure and joint
operating-characteristic calibration frozen in the E16 registration. Shared worlds
are retained in the calibration; they are not treated as independent policy samples.

The practical eligibility, retention, collateral, survival, and integrity clauses
remain the E16 registration's conjunctive conditions. A point estimate does not gain
family error merely by being described as an explanation when it is used to admit or
rank an arm.

E16's registered H5000 evaluation is one look. If a future protocol permits repeated
looks, it must specify a separate, prospective alpha-spending plan in addition to the
across-arm/claim multiplicity procedure. It may not reset the .10 family allocation,
reuse spent alpha, or turn a post-look tuning change into the same treatment. A new
treatment or a new look needs its own frozen plan and fresh evaluation bank.

## Qualification outcomes

| Condition | E13 consequence |
| --- | --- |
| Any identity input/hash, coverage, export-sidecar, or count requirement is missing | `INCOMPLETE`; no platform or artifact conclusion |
| Unexplained engine/policy semantic mismatch | Artifact is not qualified; preserve the receipt and do not auto-rerun |
| On-platform continuation differs, or state is only weights-only | No exact-resume claim; artifact durability is not qualified |
| Transfer interval exceeds either bound | Colab efficacy is not predictive of the Mac strict print; no 100M admission from that Colab claim |
| All four receipts and the frozen multiplicity plan pass | Development artifact admission only; proceed only to separately registered Mac strict/serving gates |

This rule adopts Round 4 Judgment §f (E13), Astra 08's four-receipt proposal, and
Astra 09's arms-and-looks contract. It does not amend their source studies, active
T34 work, or any protected package.
