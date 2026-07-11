// Registers the jest-dom matchers on vitest's `expect` (and their TS types),
// so tests can use toBeInTheDocument / toHaveClass / etc.
import "@testing-library/jest-dom/vitest";

// Some jsdom setups don't expose localStorage (opaque origin); provide a small
// in-memory implementation so components that persist the player name work.
if (typeof globalThis.localStorage === "undefined") {
  class LocalStorageMock {
    private store: Record<string, string> = {};
    clear() {
      this.store = {};
    }
    getItem(key: string) {
      return key in this.store ? this.store[key] : null;
    }
    setItem(key: string, value: string) {
      this.store[key] = String(value);
    }
    removeItem(key: string) {
      delete this.store[key];
    }
    key(i: number) {
      return Object.keys(this.store)[i] ?? null;
    }
    get length() {
      return Object.keys(this.store).length;
    }
  }
  Object.defineProperty(globalThis, "localStorage", { value: new LocalStorageMock() });
}
