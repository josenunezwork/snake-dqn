import { render, fireEvent } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import EgoRasterViewer from "./EgoRasterViewer";
import type { HeroRasterDTO } from "../types";

// jsdom has no 2D canvas; stub a context that records the fillRect calls so we
// can assert the viewer actually paints cells.
let fillRectCalls = 0;
beforeAll(() => {
  fillRectCalls = 0;
  // @ts-expect-error - test stub for the 2D context jsdom lacks
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(() => {
      fillRectCalls += 1;
    }),
    strokeRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
  }));
});

// A tiny 5x5 tactical raster so tests stay fast; the component reads
// tactical_size, not a hard-coded 31, so this exercises the same paths.
function sampleRaster(size = 5): HeroRasterDTO {
  const code = Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
  const value = Array.from({ length: size }, () => Array.from({ length: size }, () => 0));
  const mid = Math.floor(size / 2);
  code[mid][mid] = 8; // own_head at centre
  value[mid][mid] = 255;
  code[0][0] = 1; // wall corner
  value[0][0] = 255;
  code[mid][0] = 2; // food to the left
  value[mid][0] = 200;
  code[0][mid] = 7; // enemy head ahead
  value[0][mid] = 128;
  return {
    tactical_size: size,
    tactical_code: code,
    tactical_value: value,
    tactical_channels: [
      "own_body",
      "enemy_body",
      "enemy_head",
      "enemy_pred",
      "ambient_food",
      "corpse_food",
      "wall",
      "own_head",
      "reserved",
    ],
    tactical_codes: {
      empty: 0,
      wall: 1,
      ambient_food: 2,
      corpse_food: 3,
      enemy_pred: 4,
      own_body: 5,
      enemy_body: 6,
      enemy_head: 7,
      own_head: 8,
    },
    strategic_size: 3,
    strategic: [],
    strategic_channels: ["enemy_mass_density", "food_mass_density", "own_body_density"],
    scalars: new Array(26).fill(0),
    mask: [true, true, true, false, false, false],
  };
}

describe("EgoRasterViewer", () => {
  it("renders the canvas, a legend, and paints cells for a raster frame", () => {
    fillRectCalls = 0;
    const { container, getByText } = render(
      <EgoRasterViewer raster={sampleRaster()} obsSpec="raster31v2" />
    );
    // canvas is present and labelled
    const canvas = container.querySelector("canvas.raster-canvas");
    expect(canvas).toBeTruthy();
    // legend swatches drawn (one per type-code style)
    expect(container.querySelectorAll(".raster-swatch").length).toBeGreaterThan(0);
    // legend text for a known code is shown
    expect(getByText("Own head")).toBeTruthy();
    // the draw effect painted the backdrop + several cells + crosshair
    expect(fillRectCalls).toBeGreaterThan(1);
  });

  it("offers a Composite tab plus a channel selector", () => {
    const { getByRole } = render(
      <EgoRasterViewer raster={sampleRaster()} obsSpec="raster31v2" />
    );
    // Composite tab exists and is selected by default
    const composite = getByRole("tab", { name: "Composite" });
    expect(composite.getAttribute("aria-selected")).toBe("true");
    // a channel chip (e.g. Wall) can be selected
    const wall = getByRole("tab", { name: "Wall" });
    fireEvent.click(wall);
    expect(wall.getAttribute("aria-selected")).toBe("true");
    expect(composite.getAttribute("aria-selected")).toBe("false");
  });

  it("degrades gracefully when the policy is vector61 (no raster)", () => {
    const { queryByRole, getByText } = render(
      <EgoRasterViewer raster={null} obsSpec="vector61" />
    );
    expect(getByText("No ego-raster for this model")).toBeTruthy();
    // no canvas / channel tabs when there is nothing to show
    expect(queryByRole("tab")).toBeNull();
  });

  it("explains that the raster view is unavailable for ego2s models and recorded replays without calling ego2s a feature vector", () => {
    const { queryByRole, queryByText, getByText } = render(
      <EgoRasterViewer raster={null} obsSpec="ego2s-bv8" />
    );
    expect(getByText("Raster view unavailable")).toBeTruthy();
    expect(
      getByText(/unavailable for the served ego2s model and recorded replays/i)
    ).toBeTruthy();
    expect(queryByText(/hand-crafted feature vector/i)).toBeNull();
    expect(queryByRole("tab")).toBeNull();
  });

  it("shows a waiting placeholder when raster31v2 has no observation yet", () => {
    const { getByText, queryByRole } = render(
      <EgoRasterViewer raster={null} obsSpec="raster31v2" />
    );
    expect(getByText("No observation yet")).toBeTruthy();
    expect(queryByRole("tab")).toBeNull();
  });

  it("renders the 6-bit action mask with illegal actions struck out", () => {
    const { getByLabelText, container } = render(
      <EgoRasterViewer raster={sampleRaster()} obsSpec="raster31v2" />
    );
    // fixture mask: [true, true, true, false, false, false]
    expect(getByLabelText("L: legal")).toBeTruthy();
    expect(getByLabelText("B-L: masked")).toBeTruthy();
    expect(container.querySelectorAll("[aria-label='Legal actions'] > *").length).toBe(6);
  });

  it("renders one strategic plane canvas per served density plane", () => {
    const size = 3;
    const plane = Array.from({ length: size }, () => Array.from({ length: size }, () => 128));
    const raster = { ...sampleRaster(), strategic: [plane, plane, plane], strategic_size: size };
    const { getByLabelText } = render(<EgoRasterViewer raster={raster} obsSpec="raster31v2" />);
    expect(getByLabelText("Strategic plane Enemy Mass Density")).toBeTruthy();
    expect(getByLabelText("Strategic plane Food Mass Density")).toBeTruthy();
    expect(getByLabelText("Strategic plane Own Body Density")).toBeTruthy();
  });

  it("omits the strategic card when no planes are served", () => {
    const { queryByText } = render(
      <EgoRasterViewer raster={sampleRaster()} obsSpec="raster31v2" />
    );
    expect(queryByText(/density planes/)).toBeNull();
  });

  it("labels the 26 scalars with the featurizer group names", () => {
    const { getByText } = render(
      <EgoRasterViewer raster={sampleRaster()} obsSpec="raster31v2" />
    );
    expect(getByText("Scalars (26-D)")).toBeTruthy();
    expect(getByText("Mass rank percentile")).toBeTruthy();
    expect(getByText("Nearest food ego (dx/dy/dist)")).toBeTruthy();
  });

  it("prefers scalar group labels served in the payload over the local fallback", () => {
    const raster = {
      ...sampleRaster(),
      scalar_groups: [{ name: "Served label", start: 0, end: 2 }],
    } as ReturnType<typeof sampleRaster>;
    const { getByText, queryByText } = render(
      <EgoRasterViewer raster={raster} obsSpec="raster31v2" />
    );
    expect(getByText("Served label")).toBeTruthy();
    expect(queryByText("Mass rank percentile")).toBeNull();
  });

  it("annotates the ahead axis only when labels are enabled", () => {
    const withLabels = render(
      <EgoRasterViewer raster={sampleRaster()} obsSpec="raster31v2" labels />
    );
    expect(withLabels.container.querySelector(".raster-axis")).toBeTruthy();

    const noLabels = render(
      <EgoRasterViewer raster={sampleRaster()} obsSpec="raster31v2" labels={false} />
    );
    expect(noLabels.container.querySelector(".raster-axis")).toBeNull();
  });

  it("renders ego2s local (7 channels), global (4 channels), and 60 grouped scalars from base64 payload", () => {
    const localS = 4;
    const globalS = 3;
    const localBytes = new Uint8Array(7 * localS * localS);
    localBytes[0] = 255; // own_ttl cell
    localBytes[2 * localS * localS + 1] = 200; // enemy_head cell
    const globalBytes = new Uint8Array(4 * globalS * globalS);
    globalBytes[0] = 180; // own_mass cell
    globalBytes[3 * globalS * globalS + 2] = 255; // outside cell

    const toB64 = (arr: Uint8Array) =>
      btoa(Array.from(arr, (b) => String.fromCharCode(b)).join(""));

    const ego2sRaster: HeroRasterDTO = {
      encoding: "u8_b64",
      obs_spec: "ego2s-bv8",
      local_size: localS,
      local_head_row: 2,
      local_head_col: 2,
      local_shape: [7, localS, localS],
      local_b64: toB64(localBytes),
      local_channels: [
        "own_ttl",
        "enemy_ttl",
        "enemy_head",
        "food",
        "wall",
        "reach_time",
        "enemy_next",
      ],
      global_size: globalS,
      global_shape: [4, globalS, globalS],
      global_b64: toB64(globalBytes),
      global_channels: ["own_mass", "enemy_mass", "food_mass", "outside"],
      scalars: Array.from({ length: 60 }, (_, i) => i * 0.1),
      scalar_groups: [
        { name: "ego2s-b scalars", start: 0, end: 30 },
        { name: "v8 veto features (5 per action)", start: 30, end: 60 },
      ],
    };

    fillRectCalls = 0;
    const { getByText, getByRole, getByLabelText } = render(
      <EgoRasterViewer raster={ego2sRaster} obsSpec="ego2s-bv8" />
    );

    // 7 local channels (tabs + plane canvases + raw channel labels)
    for (const [raw, pretty] of [
      ["own_ttl", "Own Ttl"],
      ["enemy_ttl", "Enemy Ttl"],
      ["enemy_head", "Enemy Head"],
      ["food", "Food"],
      ["wall", "Wall"],
      ["reach_time", "Reach Time"],
      ["enemy_next", "Enemy Next"],
    ]) {
      expect(getByRole("tab", { name: pretty })).toBeTruthy();
      expect(getByLabelText(`Local plane ${pretty}`)).toBeTruthy();
      expect(getByText(raw)).toBeTruthy();
    }

    // 4 global channels (plane canvases + pretty and raw labels)
    for (const [raw, pretty] of [
      ["own_mass", "Own Mass"],
      ["enemy_mass", "Enemy Mass"],
      ["food_mass", "Food Mass"],
      ["outside", "Outside"],
    ]) {
      expect(getByLabelText(`Global plane ${pretty}`)).toBeTruthy();
      expect(getByText(pretty)).toBeTruthy();
      expect(getByText(raw)).toBeTruthy();
    }

    // Switching local channel tabs works
    const reachTab = getByRole("tab", { name: "Reach Time" });
    fireEvent.click(reachTab);
    expect(reachTab.getAttribute("aria-selected")).toBe("true");

    // 60 scalars grouped as server labels them
    expect(getByText("Scalars (60-D)")).toBeTruthy();
    expect(getByText("ego2s-b scalars")).toBeTruthy();
    expect(getByText("v8 veto features (5 per action)")).toBeTruthy();
    expect(fillRectCalls).toBeGreaterThan(3);
  });
});
