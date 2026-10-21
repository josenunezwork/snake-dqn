# Two-expert panel review (2026-10-14/15)

A read-only design discussion of the B8 gap. Two panelists took part: one on game mechanics and
theory, one on deep RL. A chair wrote the synthesis, and a librarian answered factual questions.
Nothing was trained or modified. The memos cite the repository as of Oct 8. Their claims about the
Mac trainer `train_m3c.py` are inferred, and the memos mark them as inferred.

**Cited by:** [panel_review_summary_2026-10-14.md](../panel_review_summary_2026-10-14.md),
[INDEX.md](../INDEX.md), and three specs:
[t1a_commitment](../specs/t1a_commitment_spec_2026-10-15.md),
[t2_probe](../specs/t2_probe_spec_2026-10-15.md) and
[t3_rollout_labels](../specs/t3_rollout_labels_spec_2026-10-15.md).

**What it supports:**

- The panel's diagnosis: B8's remaining deaths are late neck-sealing enclosures that the learner
  cannot value, not a missing veto. Three causes are named: the Huber loss fits a median, the
  margin anchor keeps the current argmax, and replay content cycles with the segment clock.
- The recommended order: measure first (T1), then a learner-regime screen (T2), then
  rollout-labelled training (T3).

| File | Contents |
| --- | --- |
| [PANEL_DECISION_MEMO.md](PANEL_DECISION_MEMO.md) | Chair's synthesis: bottom line, shared diagnosis, open disagreements, first three experiments, owner decisions, verified code facts. Start here. |
| [round1_mechanics.md](round1_mechanics.md) | Round 1 memo from the mechanics / game-theory panelist. |
| [round1_rl.md](round1_rl.md) | Round 1 memo from the deep RL panelist. |
| [round2_mechanics.md](round2_mechanics.md) | Round 2 mechanics memo, written after reading the RL memo. |
| [round2_rl.md](round2_rl.md) | Round 2 RL memo, written after reading and checking the mechanics memo. |
| [librarian_answers.md](librarian_answers.md) | Factual answers drawn from the Mac-session report, the digests and the repo; derived arithmetic is marked. |
