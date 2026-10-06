import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import SteeringWheel, { confidence } from "./SteeringWheel";

const LABELS = ["Left", "Straight", "Right", "Boost L", "Boost S", "Boost R"];

describe("confidence()", () => {
  it("buckets the relative decision margin (raw margin / Q spread)", () => {
    expect(confidence(0.6).label).toBe("confident");
    expect(confidence(0.15).label).toBe("moderate");
    expect(confidence(0.05).label).toBe("close call");
  });
});

describe("SteeringWheel", () => {
  it("renders six action wedges and glows exactly the chosen one", () => {
    const q = [10, 20, 30, 5, 6, 7];
    const { container } = render(<SteeringWheel q={q} chosen={2} labels={LABELS} />);
    const wedges = container.querySelectorAll("path");
    expect(wedges.length).toBe(6);
    const glowing = [...wedges].filter((p) => p.getAttribute("filter"));
    expect(glowing.length).toBe(1);
    expect(glowing[0].querySelector("title")?.textContent).toBe("Right: 30.00");
  });

  it("marks the greedy pick separately when it differs from the action taken", () => {
    const q = [10, 20, 30, 5, 6, 7];
    // action taken = 0 (exploration), greedy argmax = 2
    const { container, getByText } = render(
      <SteeringWheel q={q} chosen={0} greedy={2} labels={LABELS} />
    );
    const dashed = [...container.querySelectorAll("path")].filter((p) =>
      p.getAttribute("stroke-dasharray")
    );
    expect(dashed.length).toBe(1);
    expect(dashed[0].querySelector("title")?.textContent).toContain("greedy pick");
    expect(getByText(/greedy pick: Right/)).toBeInTheDocument();
    // the glowing wedge is still the action taken
    const glowing = [...container.querySelectorAll("path")].filter((p) => p.getAttribute("filter"));
    expect(glowing[0].querySelector("title")?.textContent).toContain("Left");
  });

  it("adds no greedy annotations when greedy matches or is absent", () => {
    const { container } = render(
      <SteeringWheel q={[9, 1, 1, 1, 1, 1]} chosen={0} greedy={0} labels={LABELS} />
    );
    expect(
      [...container.querySelectorAll("path")].filter((p) => p.getAttribute("stroke-dasharray"))
        .length
    ).toBe(0);
  });

  it("places the chosen wedge on the side matching its direction", () => {
    // chosen = Left (dir 0) should render to the left (avg x < 0)
    const { container } = render(
      <SteeringWheel q={[99, 1, 1, 1, 1, 1]} chosen={0} labels={LABELS} />
    );
    const chosen = [...container.querySelectorAll("path")].find((p) => p.getAttribute("filter"));
    const nums = (chosen?.getAttribute("d") || "").match(/-?\d+\.?\d*/g)?.map(Number) || [];
    // sample the move/line vertices (skip arc radius tokens is imperfect, but the
    // x-extent still leans strongly negative for a left wedge)
    const xs = nums.filter((_, i) => i % 2 === 0);
    const avgX = xs.reduce((a, b) => a + b, 0) / xs.length;
    expect(avgX).toBeLessThan(0);
  });

  it("reads the same confidence for a ~4x smaller Q scale", () => {
    // champion-scale Q and the same preferences at FRP-v3's ~1/4 scale
    const big = [48.0, 47.4, 46.5, 45.0, 44.9, 44.2];
    const small = big.map((v) => v / 4);
    const a = render(<SteeringWheel q={big} chosen={0} labels={LABELS} />);
    const b = render(<SteeringWheel q={small} chosen={0} labels={LABELS} />);
    const label = (c: HTMLElement) => c.querySelector(".wheel-conf")?.textContent?.split(" · ")[0];
    expect(label(a.container)).toBe("moderate");
    expect(label(b.container)).toBe("moderate");
    // the caption still shows the raw margin in the checkpoint's own units
    expect(b.container.querySelector(".wheel-conf")?.textContent).toContain("+0.15");
  });
});
