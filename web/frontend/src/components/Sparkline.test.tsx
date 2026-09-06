import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Sparkline from "./Sparkline";

describe("Sparkline", () => {
  it("renders an empty svg when there are fewer than two data points", () => {
    const { container } = render(<Sparkline values={[5]} />);
    const paths = container.querySelectorAll("path");
    expect(paths.length).toBe(0);
  });

  it("draws a line path for a series of values", () => {
    const { container } = render(<Sparkline values={[1, 2, 3, 4]} fill={false} />);
    const line = container.querySelector("path");
    expect(line).toBeTruthy();
    // starts with a single move, then line-tos
    expect(line?.getAttribute("d")?.startsWith("M")).toBe(true);
    expect((line?.getAttribute("d")?.match(/M/g) || []).length).toBe(1);
  });

  it("breaks the line across null gaps instead of plotting zero", () => {
    // a gap in the middle should produce two separate sub-paths (two M commands)
    const { container } = render(<Sparkline values={[1, 2, null, 4, 5]} fill={false} />);
    const d = container.querySelector("path")?.getAttribute("d") || "";
    expect((d.match(/M/g) || []).length).toBe(2);
  });

  it("shows min/max range labels only when showRange is set", () => {
    const withRange = render(<Sparkline values={[1, 2, 10]} showRange />);
    const texts = [...withRange.container.querySelectorAll(".spark-range text")].map(
      (t) => t.textContent
    );
    expect(texts).toContain("10");
    expect(texts).toContain("1");

    const without = render(<Sparkline values={[1, 2, 10]} />);
    expect(without.container.querySelector(".spark-range")).toBeNull();
  });

  it("collapses to a single range label when the window is flat", () => {
    const { container } = render(<Sparkline values={[3, 3, 3]} showRange />);
    expect(container.querySelectorAll(".spark-range text").length).toBe(1);
  });

  it("formats range labels with the provided formatter", () => {
    const { container } = render(
      <Sparkline values={[0.101, 0.103]} showRange format={(v) => v.toFixed(3)} />
    );
    const texts = [...container.querySelectorAll(".spark-range text")].map((t) => t.textContent);
    expect(texts).toContain("0.103");
    expect(texts).toContain("0.101");
  });

  it("omits the area fill when the series has gaps", () => {
    const { container } = render(<Sparkline values={[1, null, 3, 4]} fill />);
    // only the stroked line path should exist, not a filled area path
    const filled = [...container.querySelectorAll("path")].filter(
      (p) => p.getAttribute("fill") && p.getAttribute("fill") !== "none"
    );
    expect(filled.length).toBe(0);
  });
});
