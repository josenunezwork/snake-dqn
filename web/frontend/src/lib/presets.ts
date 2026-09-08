// Shared speed presets so the Controls segmented buttons, the number-key (1–5)
// shortcuts, and the command palette all agree on the same fps values. 1× is the
// engine's default 12fps; "Max" is the WebSocket-practical ceiling.
export interface SpeedPreset {
  label: string;
  fps: number;
}

export const SPEED_PRESETS: SpeedPreset[] = [
  { label: "0.5×", fps: 6 },
  { label: "1×", fps: 12 },
  { label: "2×", fps: 24 },
  { label: "4×", fps: 48 },
  { label: "Max", fps: 60 },
];

// The preset whose fps best matches the live speed (nearest), for highlighting.
export function activePresetIndex(speed: number): number {
  let best = 0;
  let bestD = Infinity;
  SPEED_PRESETS.forEach((p, i) => {
    const d = Math.abs(p.fps - speed);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}
