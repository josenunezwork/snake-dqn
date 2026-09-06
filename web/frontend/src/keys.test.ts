import { describe, expect, it } from "vitest";
import { resolveMoveKey } from "./keys";

describe("resolveMoveKey", () => {
  it("maps arrow key codes", () => {
    expect(resolveMoveKey("ArrowUp")).toBe("up");
    expect(resolveMoveKey("ArrowDown")).toBe("down");
    expect(resolveMoveKey("ArrowLeft")).toBe("left");
    expect(resolveMoveKey("ArrowRight")).toBe("right");
  });

  it("maps the physical WASD cluster via e.code (layout-independent)", () => {
    expect(resolveMoveKey("KeyW")).toBe("up");
    expect(resolveMoveKey("KeyA")).toBe("left");
    expect(resolveMoveKey("KeyS")).toBe("down");
    expect(resolveMoveKey("KeyD")).toBe("right");
  });

  it("steers on AZERTY: physical W/A positions produce z/q but the code wins", () => {
    // On AZERTY the physical QWERTY-W key reports code KeyW with key "z".
    expect(resolveMoveKey("KeyW", "z")).toBe("up");
    expect(resolveMoveKey("KeyA", "q")).toBe("left");
  });

  it("falls back to the produced character when the code is unknown", () => {
    expect(resolveMoveKey("", "w")).toBe("up");
    expect(resolveMoveKey("", "a")).toBe("left");
    expect(resolveMoveKey("", "s")).toBe("down");
    expect(resolveMoveKey("", "d")).toBe("right");
    expect(resolveMoveKey("", "ArrowUp")).toBe("up");
  });

  it("tolerates CapsLock/Shift in the key fallback", () => {
    expect(resolveMoveKey("", "W")).toBe("up");
    expect(resolveMoveKey("", "A")).toBe("left");
    expect(resolveMoveKey("", "S")).toBe("down");
    expect(resolveMoveKey("", "D")).toBe("right");
  });

  it("returns undefined for unrelated keys", () => {
    expect(resolveMoveKey("KeyQ", "q")).toBeUndefined();
    expect(resolveMoveKey("Enter", "Enter")).toBeUndefined();
    expect(resolveMoveKey("Space", " ")).toBeUndefined();
  });
});
