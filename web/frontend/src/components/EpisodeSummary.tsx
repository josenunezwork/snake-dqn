import { useEffect, useRef, useState } from "react";
import type { Stats } from "../types";

// Watch/train mode loops silently, so an episode's start and end are invisible.
// When the engine resets a game (the frame counter regresses — the same signal
// the canvas uses), we surface a brief "Episode over" card summarising the run
// that just ended and how its peak length compares to the previous one. This is
// the closure the Play tab already has, brought to the AI's own games.

interface Summary {
  bestLength: number;
  food: number;
  kills: number;
  frames: number;
  delta: number | null; // best-length change vs the previous episode
}

const HOLD_MS = 5000;
const MIN_FRAMES = 40; // ignore trivial resets (e.g. an immediate re-reset)

export default function EpisodeSummary({
  stats,
  frameNum,
  mode,
}: {
  stats: Stats | null;
  frameNum: number;
  mode: string;
}) {
  const prevStats = useRef<Stats | null>(null);
  const prevFrame = useRef<number>(0);
  const lastBest = useRef<number | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);

  useEffect(() => {
    // Play mode has its own game-over UI; don't track episodes there.
    if (mode === "play") {
      prevStats.current = null;
      prevFrame.current = 0;
      return;
    }
    const prev = prevStats.current;
    const regressed = frameNum < prevFrame.current - 5;
    if (prev && regressed && prevFrame.current >= MIN_FRAMES) {
      const delta = lastBest.current != null ? prev.best_length - lastBest.current : null;
      lastBest.current = prev.best_length;
      setSummary({
        bestLength: prev.best_length,
        food: prev.food_eaten,
        kills: prev.kills,
        frames: prevFrame.current,
        delta,
      });
    }
    prevStats.current = stats;
    prevFrame.current = frameNum;
  }, [frameNum, stats, mode]);

  useEffect(() => {
    if (!summary) return;
    const id = setTimeout(() => setSummary(null), HOLD_MS);
    return () => clearTimeout(id);
  }, [summary]);

  if (!summary) return null;
  const record = summary.delta != null && summary.delta > 0;
  return (
    <div className="episode-card" role="status" onClick={() => setSummary(null)}>
      <div className="episode-head">
        <span>Episode over</span>
        {record && <span className="episode-record">★ new record</span>}
      </div>
      <div className="episode-stats">
        <div className="es-cell">
          <div className="es-k">Best length</div>
          <div className="es-v">{summary.bestLength}</div>
        </div>
        <div className="es-cell">
          <div className="es-k">Food</div>
          <div className="es-v">{summary.food}</div>
        </div>
        <div className="es-cell">
          <div className="es-k">Kills</div>
          <div className="es-v">{summary.kills}</div>
        </div>
        <div className="es-cell">
          <div className="es-k">Frames</div>
          <div className="es-v">{summary.frames.toLocaleString()}</div>
        </div>
      </div>
      {summary.delta != null && summary.delta !== 0 && (
        <div className={"episode-delta " + (record ? "up" : "down")}>
          {record ? "▲" : "▼"} {record ? "+" : ""}
          {summary.delta} vs last episode
        </div>
      )}
    </div>
  );
}
