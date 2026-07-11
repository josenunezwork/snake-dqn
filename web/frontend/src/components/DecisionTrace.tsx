import { useEffect, useRef, useState } from "react";
import Sparkline from "./Sparkline";

// Per-action colours (0 L, 1 S, 2 R, 3 boost-L, 4 boost-S, 5 boost-R). Boost
// variants are the lighter tint of their direction hue.
export const ACTION_COLORS = ["#38bdf8", "#34d399", "#f472b6", "#7dd3fc", "#6ee7b7", "#f9a8d4"];

interface Sample {
  chosen: number;
  margin: number;
}

interface Props {
  tick: number; // frame counter — one sample appended per new frame
  chosen: number;
  margin: number;
  labels: string[];
  resetKey: string | number; // clears the trace when the model/hero identity changes
  cap?: number;
}

// A temporal read of the agent's decision: a confidence (margin) sparkline over
// the recent window, and a "piano roll" of which action it committed to each
// frame — so you can watch it dither across close calls, then lock in.
export default function DecisionTrace({ tick, chosen, margin, labels, resetKey, cap = 160 }: Props) {
  const buf = useRef<Sample[]>([]);
  const lastTick = useRef<number>(-1);
  const [, bump] = useState(0);

  // reset on identity change (new checkpoint / hero)
  useEffect(() => {
    buf.current = [];
    lastTick.current = -1;
    bump((n) => n + 1);
  }, [resetKey]);

  // one sample per distinct frame
  useEffect(() => {
    if (tick === lastTick.current) return;
    lastTick.current = tick;
    const arr = buf.current;
    arr.push({ chosen, margin });
    if (arr.length > cap) arr.splice(0, arr.length - cap);
    bump((n) => n + 1);
  }, [tick, chosen, margin, cap]);

  const samples = buf.current;
  if (samples.length < 2) {
    return (
      <div className="trace-empty muted">Collecting decision history…</div>
    );
  }

  const margins = samples.map((s) => s.margin);
  const W = 100;
  const H = 16;
  const cw = W / cap;

  return (
    <div className="trace">
      <div className="trace-head">
        <span className="trace-label">Confidence</span>
        <span className="muted mono" style={{ fontSize: 10 }}>
          last {samples.length}f
        </span>
      </div>
      <Sparkline values={margins} color="#fbbf24" width={100} height={26} />
      <div className="trace-head" style={{ marginTop: 6 }}>
        <span className="trace-label">Chosen action</span>
      </div>
      <svg className="piano" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="chosen action over time">
        {samples.map((s, i) => (
          <rect
            key={i}
            x={(cap - samples.length + i) * cw}
            y={0}
            width={Math.max(cw, 0.4)}
            height={H}
            fill={ACTION_COLORS[s.chosen] ?? "#8a97a8"}
            opacity={0.9}
          >
            <title>{labels[s.chosen] ?? "?"}</title>
          </rect>
        ))}
      </svg>
      <div className="trace-legend">
        {labels.map((l, i) => (
          <span key={i} className="trace-key">
            <span className="trace-dot" style={{ background: ACTION_COLORS[i] }} />
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}
