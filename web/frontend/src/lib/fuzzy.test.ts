import { describe, expect, it } from "vitest";
import { fuzzyScore, fuzzyFilter } from "./fuzzy";

describe("fuzzyScore", () => {
  it("returns 0 for an empty query (everything matches)", () => {
    expect(fuzzyScore("", "anything")).toBe(0);
  });

  it("returns null when a char is missing", () => {
    expect(fuzzyScore("xyz", "Load checkpoint")).toBeNull();
  });

  it("matches a subsequence in order", () => {
    expect(fuzzyScore("ldck", "Load checkpoint")).not.toBeNull();
  });

  it("scores a contiguous prefix higher than a scattered match", () => {
    const contig = fuzzyScore("load", "Load checkpoint")!;
    const scattered = fuzzyScore("load", "l o a d rummage")!;
    expect(contig).toBeGreaterThan(scattered);
  });
});

describe("fuzzyFilter", () => {
  const items = ["Play", "Pause", "Reset game", "Load champion", "Train mode"];
  const id = (s: string) => s;

  it("keeps original order for an empty query", () => {
    expect(fuzzyFilter(items, "", id)).toEqual(items);
  });

  it("ranks the best match first", () => {
    const out = fuzzyFilter(items, "train", id);
    expect(out[0]).toBe("Train mode");
  });

  it("drops non-matches", () => {
    const out = fuzzyFilter(items, "zzz", id);
    expect(out).toEqual([]);
  });
});
