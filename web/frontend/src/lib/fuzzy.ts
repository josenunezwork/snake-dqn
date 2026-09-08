// Tiny dependency-free fuzzy matcher for the command palette. Subsequence match
// (all query chars appear in order) with a score that rewards contiguous runs,
// word-boundary starts, and early matches — so "ld chk" ranks "Load checkpoint"
// above an incidental scatter match.

export function fuzzyScore(query: string, text: string): number | null {
  const q = query.trim().toLowerCase();
  if (!q) return 0;
  const t = text.toLowerCase();
  let ti = 0;
  let score = 0;
  let run = 0;
  let prevMatchIdx = -1;
  for (let qi = 0; qi < q.length; qi++) {
    const ch = q[qi];
    const found = t.indexOf(ch, ti);
    if (found === -1) return null;
    // base point for the match
    score += 1;
    // contiguous run bonus
    if (found === prevMatchIdx + 1) {
      run += 1;
      score += run * 3;
    } else {
      run = 0;
    }
    // word-boundary bonus (start of string or after space/·/-)
    if (found === 0 || /[\s·\-/]/.test(t[found - 1])) score += 4;
    // small penalty for the gap we skipped
    score -= Math.min(found - ti, 4) * 0.2;
    prevMatchIdx = found;
    ti = found + 1;
  }
  return score;
}

export interface Scored<T> {
  item: T;
  score: number;
}

// Filter + sort by relevance. With an empty query every item passes (score 0),
// preserving the caller's original order.
export function fuzzyFilter<T>(items: T[], query: string, key: (t: T) => string): T[] {
  if (!query.trim()) return items;
  const scored: Scored<T>[] = [];
  items.forEach((item, i) => {
    const s = fuzzyScore(query, key(item));
    if (s !== null) scored.push({ item, score: s - i * 0.001 });
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.map((s) => s.item);
}
