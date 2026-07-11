import { describe, expect, it } from "vitest";
import { resolveMoveKey } from "./keys";

describe("resolveMoveKey", () => {
  it("maps arrow keys", () => {
    expect(resolveMoveKey("ArrowUp")).toBe("up");
    expect(resolveMoveKey("ArrowDown")).toBe("down");
    expect(resolveMoveKey("ArrowLeft")).toBe("left");
    expect(resolveMoveKey("ArrowRight")).toBe("right");
  });

  it("maps lowercase WASD", () => {
    expect(resolveMoveKey("w")).toBe("up");
    expect(resolveMoveKey("a")).toBe("left");
    expect(resolveMoveKey("s")).toBe("down");
    expect(resolveMoveKey("d")).toBe("right");
  });

  it("maps uppercase WASD (CapsLock on / Shift held)", () => {
    expect(resolveMoveKey("W")).toBe("up");
    expect(resolveMoveKey("A")).toBe("left");
    expect(resolveMoveKey("S")).toBe("down");
    expect(resolveMoveKey("D")).toBe("right");
  });

  it("returns undefined for unrelated keys", () => {
    expect(resolveMoveKey("q")).toBeUndefined();
    expect(resolveMoveKey("Enter")).toBeUndefined();
    expect(resolveMoveKey(" ")).toBeUndefined();
  });
});
