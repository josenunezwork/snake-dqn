# Follow-up for the speed advisors (2026-10-19 ~21:00 PT): better GPUs?

The owner asks: **can a better GPU be used?** Context since your first answers: all four speed advisors
converged on a deterministic shadow trajectory (one greedy continuation kept 300-500 frames ahead, verify
labels free) plus expanding once per doomed trajectory, bisecting toward the last avoidable decision.
Your answers: docs/research/advice/2026-10-19_t34_speed_astra_{perf,search}.md (main) and
/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-speed-advice/fable_{systems,search}.md.

FACTS. Owner's Colab Pro+ (~1,160 prepaid units) offers: L4 (~1.7 u/h, plan with 2.0; 12 vCPU),
A100-40GB ~5.4 u/h, A100-80GB ~7.5 u/h, H100 (scarce, rate unmeasured, > A100-80), G4 = RTX PRO 6000
Blackwell ~96 GB ~8.7 u/h (needs torch cu128). vCPU counts on the bigger runtimes: check
~/.claude/skills/colab/SKILL.md and ~/.claude/skills/colab/*.md and the colab_cli notes; say if unknown.
RunPod is available but disfavoured (owner prefers Colab). The live Watch hero (Path B) runs on the
owner's Mac (Apple Silicon; MPS ~5x slower than CPU for tiny models in past tests). The training teacher
and student fits run on Colab. B8 = ego2s-bv8 dueling CNN, ~1.64M params; opponents = frozen vector61 MLPs
with v8 veto, or scripted bots; simulator = numba GridBatchSim on host CPU.

OPEN QUESTION. Under the shadow design, where (if anywhere) would a better GPU — or a different kind of
hardware — actually change the teacher's or the fits' time or cost, and what would you run it on and how?
Reason from where the work is and what each runtime gives you (GPU, host cores, RAM, units/h); include
units-per-useful-result estimates and what to measure first. Same RULES as QUESTION.md. <=700 words.
