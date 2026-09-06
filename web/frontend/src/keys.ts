// Arrow / WASD keys -> the named directions the backend's human input expects.
// Kept standalone so it can be unit-tested without mounting the whole App.
//
// Steering matches on KeyboardEvent.code (the PHYSICAL key) first, so the WASD
// cluster works on AZERTY/Dvorak and other non-QWERTY layouts; arrows share the
// same code and key name. A key-based fallback keeps layouts whose keys are
// *labelled* W/A/S/D (but sit on other physical positions) working too.
export const MOVE_CODES: Record<string, string> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
  KeyW: "up",
  KeyS: "down",
  KeyA: "left",
  KeyD: "right",
};

// Fallback on the produced character (lowercased so CapsLock/Shift don't
// matter), plus arrows for synthetic events that carry no code.
const MOVE_FALLBACK: Record<string, string> = {
  arrowup: "up",
  arrowdown: "down",
  arrowleft: "left",
  arrowright: "right",
  w: "up",
  s: "down",
  a: "left",
  d: "right",
};

// Resolve a keydown to a movement direction: physical code first, then key.
export function resolveMoveKey(code: string, key = ""): string | undefined {
  return MOVE_CODES[code] ?? MOVE_FALLBACK[key.toLowerCase()];
}
