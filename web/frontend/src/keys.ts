// Arrow / WASD keys -> the named directions the backend's human input expects.
// Kept standalone so it can be unit-tested without mounting the whole App.
export const MOVE_KEYS: Record<string, string> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
  w: "up",
  s: "down",
  a: "left",
  d: "right",
};

// Resolve a KeyboardEvent.key to a movement direction, tolerating CapsLock/Shift.
// Arrow keys (case-invariant) resolve directly; letters fall back to lowercase so
// "W"/"A"/"S"/"D" (CapsLock on, or Shift held) still steer.
export function resolveMoveKey(key: string): string | undefined {
  return MOVE_KEYS[key] ?? MOVE_KEYS[key.toLowerCase()];
}
