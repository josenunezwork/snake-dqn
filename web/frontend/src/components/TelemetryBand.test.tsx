import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import TelemetryBand from "./TelemetryBand";
import type { MetricSample } from "../types";

const sample = (over: Partial<MetricSample>): MetricSample => ({
  frame: 0,
  loss: null,
  epsilon: 0,
  foodEaten: 0,
  alive: 0,
  bestLength: 0,
  kills: 0,
  ...over,
});

describe("TelemetryBand", () => {
  it("renders nothing until there are at least two samples", () => {
    const { container } = render(<TelemetryBand history={[sample({ frame: 1 })]} />);
    expect(container.querySelector(".telemetry")).toBeNull();
  });

  it("renders a tile per metric with the latest value", () => {
    const history = [
      sample({ frame: 1, bestLength: 10, alive: 6, foodEaten: 100, kills: 2 }),
      sample({ frame: 2, bestLength: 42, alive: 5, foodEaten: 250, kills: 3 }),
    ];
    const { container, getAllByText } = render(<TelemetryBand history={history} />);
    expect(container.querySelectorAll(".tele-tile").length).toBe(6);
    // value appears in the tile (and again as the sparkline's max range label)
    expect(getAllByText("42").length).toBeGreaterThanOrEqual(1); // best length
    expect(getAllByText("250").length).toBeGreaterThanOrEqual(1); // food (not compacted)
  });

  it("annotates each sparkline with its window min/max range labels", () => {
    const history = [
      sample({ frame: 1, bestLength: 10 }),
      sample({ frame: 2, bestLength: 42 }),
    ];
    const { container } = render(<TelemetryBand history={history} />);
    // every tile with data gets a range; the loss tile is all-null here, so 5
    const ranges = container.querySelectorAll(".spark-range");
    expect(ranges.length).toBe(5);
    // best-length tile shows both ends of its auto-scaled window
    const texts = [...container.querySelectorAll(".spark-range text")].map((t) => t.textContent);
    expect(texts).toContain("42");
    expect(texts).toContain("10");
  });

  it("explains the jargon tiles with glossary info dots", () => {
    const history = [sample({ frame: 1 }), sample({ frame: 2 })];
    const { container } = render(<TelemetryBand history={history} />);
    // every tile carries an ⓘ (epsilon/loss/bestlen/kills/alive/food)
    expect(container.querySelectorAll(".infodot").length).toBe(6);
  });

  it("skips duplicate frames (paused heartbeats) instead of plotting repeats", () => {
    const history = [
      sample({ frame: 1, bestLength: 10 }),
      sample({ frame: 2, bestLength: 42 }),
      sample({ frame: 2, bestLength: 42 }), // paused heartbeat: frame not advanced
      sample({ frame: 2, bestLength: 42 }),
    ];
    const { container } = render(<TelemetryBand history={history} paused />);
    expect(container.querySelector(".telemetry")?.getAttribute("data-paused")).toBe("true");
    // only 2 distinct frames survive → the line path has exactly 2 points (one M, one L)
    const spark = container.querySelector(".tele-tile .spark path[fill='none']");
    const d = spark?.getAttribute("d") ?? "";
    expect((d.match(/[ML]/g) || []).length).toBe(2);
  });

  it("renders nothing when all frames are duplicates of one frame", () => {
    const history = [sample({ frame: 5 }), sample({ frame: 5 }), sample({ frame: 5 })];
    const { container } = render(<TelemetryBand history={history} />);
    expect(container.querySelector(".telemetry")).toBeNull();
  });

  it("shows an em dash for loss when it is never present", () => {
    const history = [sample({ frame: 1 }), sample({ frame: 2 })];
    const { getAllByText } = render(<TelemetryBand history={history} />);
    // loss (and, given zeros, nothing else forces it) renders the placeholder
    expect(getAllByText("—").length).toBeGreaterThanOrEqual(1);
  });

  it("compacts large food counts to k", () => {
    const history = [
      sample({ frame: 1, foodEaten: 900 }),
      sample({ frame: 2, foodEaten: 4200 }),
    ];
    const { getAllByText } = render(<TelemetryBand history={history} />);
    expect(getAllByText("4.2k").length).toBeGreaterThanOrEqual(1);
  });
});
