import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getClientId,
  getStoredPlayerName,
  normalizePlayerName,
  setStoredPlayerName,
} from "./identity";

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("player identity helpers", () => {
  it("normalizes names the same way before persistence and submission", () => {
    expect(normalizePlayerName("  Ada\t\nLovelace  ")).toBe("Ada Lovelace");
  });

  it("treats unavailable storage as an optional convenience", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });

    expect(getStoredPlayerName()).toBe("");
    expect(() => setStoredPlayerName("Ada")).not.toThrow();
    expect(getClientId()).toBe(getClientId());
  });
});
