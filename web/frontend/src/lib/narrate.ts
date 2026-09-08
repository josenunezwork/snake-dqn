import type { InspectorDTO } from "../types";

// Turn the hero's raw decision into one plain-English sentence — the kind of
// caption that closes the gap from "pretty bars" to "oh, it turned right because
// food is there and the left is a wall". Pure + deterministic so it's unit-tested
// and can be debounced by the caller.
//
// Obs-spec aware: the vector61 feature indices (food dist at [7], per-action
// danger at [54..56]) are ONLY valid for vector specs. For raster31v2 the
// inspector state is the 26-D featurizer scalar band, where index 7 is the
// hero's mass-rank percentile — reading it as food distance fabricates
// "closing on food" captions. The raster scalars' nearest-food entries
// (dx/dy/dist at 12–14) only describe food BEYOND the tactical raster
// (zeros = none out there), so they cannot honestly caption nearby food
// either; for raster models we narrate direction + confidence only.

export type Tone = "good" | "warn" | "neutral";
export interface Narration {
  text: string;
  tone: Tone;
}

const DIR_WORD = ["left", "straight", "right"];

function confidenceBucket(margin: number): { label: string; close: boolean } {
  if (margin > 1) return { label: "confident", close: false };
  if (margin > 0.3) return { label: "weighing it up", close: false };
  return { label: "a close call", close: true };
}

export function narrate(inspector: InspectorDTO | null, obsSpec?: string): Narration | null {
  if (!inspector) return null;
  const { q_values, chosen, state } = inspector;
  if (!q_values || q_values.length < 3 || chosen < 0) return null;

  // The action the hero actually took last step (post-masking / exploration),
  // when the backend serves it; fall back to the greedy argmax.
  const executed = inspector.executed_action;
  const acted =
    typeof executed === "number" && executed >= 0 && executed < q_values.length
      ? executed
      : chosen;

  const dir = acted % 3; // 0 L, 1 S, 2 R
  const boost = acted >= 3;
  const verb = boost
    ? `Boosting ${DIR_WORD[dir]}`
    : dir === 1
      ? "Going straight"
      : `Turning ${DIR_WORD[dir]}`;

  // decision margin (confidence)
  const sorted = [...q_values].sort((a, b) => b - a);
  const margin = sorted.length > 1 ? sorted[0] - sorted[1] : 0;
  const conf = confidenceBucket(margin);

  // Vector-spec detection: trust the served obs_spec when present; otherwise
  // infer from the state length (vector states are 58/61-D, raster scalars 26-D).
  const isVector = obsSpec ? obsSpec.startsWith("vector") : state.length >= 58;

  // Per-action danger [54..56] (L/S/R) and free-space (58..60) — vector61 only.
  const danger = isVector && state.length > 56 ? [state[54], state[55], state[56]] : null;
  const free =
    inspector.free_space && inspector.free_space.length >= 3 ? inspector.free_space : null;
  const foodDist = isVector && state.length > 7 ? state[7] : 1;

  let clause = "";
  let tone: Tone = "neutral";

  if (free && free[dir] < 0.34) {
    clause = "into tight space";
    tone = "warn";
  } else if (danger) {
    // Is the agent steering away from a clearly more dangerous direction?
    const chosenDanger = danger[dir];
    let worst = -1;
    let worstVal = chosenDanger + 0.25; // require a meaningful gap
    danger.forEach((d, i) => {
      if (i !== dir && d > worstVal) {
        worstVal = d;
        worst = i;
      }
    });
    if (worst >= 0) {
      clause = `away from danger on the ${DIR_WORD[worst]}`;
      tone = "neutral";
    }
  }

  if (!clause) {
    if (foodDist < 0.12) {
      clause = "toward food just ahead";
      tone = "good";
    } else if (foodDist < 0.3) {
      clause = "closing on food";
      tone = "good";
    }
  }

  if (conf.close) tone = "warn";

  const text = clause ? `${verb} — ${clause} · ${conf.label}` : `${verb} · ${conf.label}`;
  return { text, tone };
}
