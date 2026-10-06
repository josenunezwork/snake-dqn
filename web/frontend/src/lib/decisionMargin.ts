// Scale-free decision confidence for the Q-value read-outs (steering-wheel hub,
// narrator).
//
// The raw margin (best Q minus second-best Q) is in the checkpoint's reward
// units, so fixed cutoffs on it only fit one checkpoint. The FRP-v3 checkpoint
// was trained after a 0.25 head/reward rescale and its Q-values are about 4x
// smaller than the champion's: under the old fixed cutoffs (confident > 1,
// moderate > 0.3) about 97% of its decisions read as "close call".
//
// The relative margin divides the raw margin by the spread of the same Q vector
// (best minus worst), so it is invariant to any positive scale or offset of the
// Q-values and lies in [0, 1]. The cutoffs below were calibrated on the served
// champion (1500 Watch frames) to reproduce its old bucket split
// (~14% confident / ~44% moderate / ~42% close call); the FRP-v3 checkpoint
// gives a similar split under them (~21% / ~45% / ~34%).

export const RELATIVE_CONFIDENT = 0.25;
export const RELATIVE_MODERATE = 0.1;

export interface DecisionMargin {
  // best - second-best Q, in the checkpoint's own units (display only)
  raw: number;
  // raw / (best - worst), in [0, 1]; 0 when all Q-values are equal
  relative: number;
}

export function decisionMargin(q: readonly number[]): DecisionMargin {
  const finite = q.filter((v) => Number.isFinite(v));
  if (finite.length < 2) return { raw: 0, relative: 0 };
  const sorted = [...finite].sort((a, b) => b - a);
  const raw = sorted[0] - sorted[1];
  const spread = sorted[0] - sorted[sorted.length - 1];
  const relative = spread > 1e-9 ? raw / spread : 0;
  return { raw, relative };
}

export type ConfidenceLevel = "confident" | "moderate" | "close";

export function confidenceLevel(relative: number): ConfidenceLevel {
  if (relative > RELATIVE_CONFIDENT) return "confident";
  if (relative > RELATIVE_MODERATE) return "moderate";
  return "close";
}
