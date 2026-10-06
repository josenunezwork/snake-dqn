import { describe, expect, it } from "vitest";
import { confidenceLevel, decisionMargin, RELATIVE_CONFIDENT, RELATIVE_MODERATE } from "./decisionMargin";

describe("decisionMargin()", () => {
  it("returns the raw margin and the margin relative to the Q spread", () => {
    const m = decisionMargin([5, 1, 1, 0, 0, 0]);
    expect(m.raw).toBeCloseTo(4);
    expect(m.relative).toBeCloseTo(0.8);
  });

  it("is invariant to positive scale and offset of the Q-values", () => {
    const q = [9.3, 9.1, 8.7, 8.2, 8.0, 7.9];
    const r = decisionMargin(q).relative;
    expect(decisionMargin(q.map((v) => v * 4)).relative).toBeCloseTo(r);
    expect(decisionMargin(q.map((v) => v / 4 - 100)).relative).toBeCloseTo(r);
  });

  it("is 0 for degenerate inputs (all equal, one value, non-finite)", () => {
    expect(decisionMargin([2, 2, 2, 2, 2, 2]).relative).toBe(0);
    expect(decisionMargin([3]).relative).toBe(0);
    expect(decisionMargin([]).raw).toBe(0);
    expect(decisionMargin([NaN, 1, 0]).relative).toBeCloseTo(1);
  });
});

describe("confidenceLevel()", () => {
  it("buckets at the calibrated cutoffs (exclusive)", () => {
    expect(confidenceLevel(RELATIVE_CONFIDENT + 0.01)).toBe("confident");
    expect(confidenceLevel(RELATIVE_CONFIDENT)).toBe("moderate");
    expect(confidenceLevel(RELATIVE_MODERATE + 0.01)).toBe("moderate");
    expect(confidenceLevel(RELATIVE_MODERATE)).toBe("close");
    expect(confidenceLevel(0)).toBe("close");
  });
});
