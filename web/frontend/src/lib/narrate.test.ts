import { describe, expect, it } from "vitest";
import { narrate } from "./narrate";
import type { InspectorDTO } from "../types";

function mk(partial: Partial<InspectorDTO>): InspectorDTO {
  return {
    input_size: 61,
    state: new Array(61).fill(0),
    groups: [],
    q_values: [1, 0, 0, 0, 0, 0],
    action_labels: ["Left", "Straight", "Right", "Boost L", "Boost S", "Boost R"],
    chosen: 0,
    free_space: null,
    ...partial,
  };
}

describe("narrate()", () => {
  it("returns null without an inspector", () => {
    expect(narrate(null)).toBeNull();
  });

  it("names the move and confidence from the Q margin", () => {
    const n = narrate(mk({ q_values: [5, 1, 1, 0, 0, 0], chosen: 0 }));
    expect(n?.text.startsWith("Turning left")).toBe(true);
    expect(n?.text).toContain("confident");
  });

  it("flags a close call as a warning", () => {
    const n = narrate(mk({ q_values: [2.0, 1.95, 1, 0, 0, 0], chosen: 0 }));
    expect(n?.text).toContain("close call");
    expect(n?.tone).toBe("warn");
  });

  it("describes boosting", () => {
    const n = narrate(mk({ q_values: [0, 0, 0, 9, 1, 1], chosen: 3 }));
    expect(n?.text.startsWith("Boosting left")).toBe(true);
  });

  it("mentions food when it is close", () => {
    const state = new Array(61).fill(0);
    state[7] = 0.08; // very close nearest-food distance
    const n = narrate(mk({ state, q_values: [5, 1, 1, 0, 0, 0], chosen: 1 }));
    expect(n?.text).toContain("food");
    expect(n?.tone).toBe("good");
  });

  it("warns when the chosen move goes into tight free-space", () => {
    const n = narrate(mk({ q_values: [5, 1, 1, 0, 0, 0], chosen: 0, free_space: [0.1, 1, 1] }));
    expect(n?.text).toContain("tight");
    expect(n?.tone).toBe("warn");
  });

  it("never reads raster31v2 scalars as vector food distance (index 7 is mass rank)", () => {
    // 26-D raster scalar band; index 7 = mass rank percentile. A bottom-ranked
    // hero must NOT be narrated as "closing on food".
    const state = new Array(26).fill(0);
    state[7] = 0.08;
    const n = narrate(
      mk({ input_size: 26, state, q_values: [5, 1, 1, 0, 0, 0], chosen: 1 }),
      "raster31v2"
    );
    expect(n?.text).not.toContain("food");
    expect(n?.tone).not.toBe("good");
    expect(n?.text.startsWith("Going straight")).toBe(true);
  });

  it("infers a raster state from its length when no obs_spec is given", () => {
    const state = new Array(26).fill(0);
    state[7] = 0.08; // mass rank, would fake "food just ahead" under vector indices
    const n = narrate(mk({ input_size: 26, state, q_values: [5, 1, 1, 0, 0, 0], chosen: 1 }));
    expect(n?.text).not.toContain("food");
  });

  it("still narrates food for vector61 when the spec is passed", () => {
    const state = new Array(61).fill(0);
    state[7] = 0.08;
    const n = narrate(mk({ state, q_values: [5, 1, 1, 0, 0, 0], chosen: 1 }), "vector61");
    expect(n?.text).toContain("food");
    expect(n?.tone).toBe("good");
  });

  it("narrates the executed action when it differs from the greedy argmax", () => {
    const insp = {
      ...mk({ q_values: [5, 1, 1, 0, 0, 0], chosen: 0 }),
      executed_action: 2,
    };
    const n = narrate(insp);
    expect(n?.text.startsWith("Turning right")).toBe(true);
  });
});
