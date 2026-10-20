# Rules audit (owner request, 2026-10-19 ~22:45 PT)

The owner asks: **"Are we doing anything from old docs or rules that may not be the best for us anymore?"**

GOAL TODAY. Ship a clearly stronger Watch hero than bare B8 s3 (served since 2026-10-17) under the
owner-ratified T34 plan, fast and cheaply: docs/research/final_run_20261019/{FINAL_PLAN.md,
SUCCESS_CONTRACT.md, DESIGN_V2_SHADOW.md} on main. Resources: owner's Mac (Apple Silicon, on AC), Colab Pro+
(~1,160 prepaid units, via the CLI supervisor), RunPod (disfavoured), Sol (Codex gpt-6.1-sol) implements,
Claude plans/operates, Astra/Fable advise. The owner's time and attention are the scarcest resource.

WHERE THE RULES LIVE (read broadly; these accumulated since June 2026 and many were written for situations
that no longer exist):
- Repo: AGENTS.md, CLAUDE.md, docs/STATUS.md, docs/*.md, docs/research/** (preregistrations, gates,
  campaign retrospectives), research/colab_cli/{ACCEPTANCE,ROBUSTNESS}.md, Makefile/CI/pre-commit.
- Claude's persistent memory (rules/decisions it follows every session):
  /Users/josenunez/.claude/projects/-Users-josenunez-Projects-ml-snake-dqn/memory/ (MEMORY.md is the index;
  `feedback` and `project` files hold standing rules and decisions).
- Claude skills: /Users/josenunez/.claude/skills/*/SKILL.md (colab, colab-jobs, runpod*, speed-review);
  hooks /Users/josenunez/.claude/hooks/; Codex skills: /Users/josenunez/.codex/skills/*.
- Launch helper: /Users/josenunez/.claude/bin/codex-handoff.sh.

OPEN QUESTION. Which of our rules, standing decisions, gates, processes, defaults and habits are no longer
serving the goal (stale facts, superseded decisions, overheads that cost more than they protect,
contradictions between sources, or safety rules that are now under-protective), and what should each become?
For every finding give: the rule and its source (file + line/quote), why it was adopted, why it no longer
fits (evidence), the cost of keeping it, the risk of changing it, and a concrete replacement (keep / amend /
retire / merge). Rank by impact on the goal. Also name rules that look old but must be kept, and why.

RULES. Advice only: read files, no edits except your one answer file; no jobs, tmux, Colab, RunPod or paid
services; do not disturb the running Sol jobs, Phase 0 or S1. Answer <=1,500 words.
