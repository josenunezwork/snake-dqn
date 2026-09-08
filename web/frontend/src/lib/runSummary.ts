import type { Frame } from "../types";

// A shareable plain-text snapshot of the current run — a Play score card, or the
// live watch/train stats — for pasting into a training log or a message.
export function runSummary(frame: Frame | null): string {
  if (!frame) return "snake-dqn — (not connected)";
  const s = frame.session;
  const st = frame.stats;
  const p = frame.play;
  const lines: string[] = [];

  if (p && p.active) {
    lines.push("snake-dqn — Play run");
    lines.push(`score: ${p.score}`);
    lines.push(`length: ${p.length}`);
    lines.push(`food: ${p.food_eaten}`);
    lines.push(`kills: ${p.kills}`);
    lines.push(`frames: ${p.frames}`);
  } else if (st) {
    lines.push(`snake-dqn — ${s?.mode ?? "watch"} run`);
    lines.push(`best length: ${st.best_length}`);
    lines.push(`alive: ${st.alive}`);
    lines.push(`food eaten: ${st.food_eaten}`);
    lines.push(`kills: ${st.kills}`);
    lines.push(`deaths: ${st.deaths}`);
    lines.push(`frame: ${st.frame}`);
    if (st.loss != null) lines.push(`loss: ${st.loss.toFixed(4)}`);
    lines.push(`epsilon: ${st.epsilon.toFixed(3)}`);
  }

  if (s) {
    lines.push(`checkpoint: ${s.checkpoint ?? "none"}`);
    lines.push(`config: ${s.config}`);
    lines.push(`input: ${s.input_size}-D · snakes: ${s.num_snakes}`);
  }
  return lines.join("\n");
}

// Copy text to the clipboard with a legacy fallback for non-secure contexts.
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
