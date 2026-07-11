import type { MetricSample } from "../types";
import Sparkline from "./Sparkline";

interface Tile {
  key: string;
  label: string;
  color: string;
  get: (s: MetricSample) => number | null;
  fmt: (v: number) => string;
}

const compact = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v)));

const TILES: Tile[] = [
  { key: "best", label: "Best len", color: "#38bdf8", get: (s) => s.bestLength, fmt: (v) => String(Math.round(v)) },
  { key: "alive", label: "Alive", color: "#34d399", get: (s) => s.alive, fmt: (v) => String(Math.round(v)) },
  { key: "food", label: "Food", color: "#f59e0b", get: (s) => s.foodEaten, fmt: compact },
  { key: "kills", label: "Kills", color: "#f472b6", get: (s) => s.kills, fmt: (v) => String(Math.round(v)) },
  { key: "loss", label: "Loss", color: "#f87171", get: (s) => s.loss, fmt: (v) => v.toFixed(3) },
  { key: "eps", label: "ε", color: "#a78bfa", get: (s) => s.epsilon, fmt: (v) => v.toFixed(2) },
];

function lastNonNull(vals: (number | null)[]): number | null {
  for (let i = vals.length - 1; i >= 0; i--) if (vals[i] != null) return vals[i];
  return null;
}

// Always-visible "telemetry terminal" strip: a rolling sparkline per key live
// metric. Reads the ring buffer from useGameSocket; renders nothing until data
// arrives so it never flashes empty.
export default function TelemetryBand({ history }: { history: MetricSample[] }) {
  if (history.length < 2) return null;
  return (
    <div className="telemetry" role="group" aria-label="Live telemetry">
      {TILES.map((t) => {
        const vals = history.map(t.get);
        const cur = lastNonNull(vals);
        return (
          <div className="tele-tile" key={t.key}>
            <div className="tele-head">
              <span className="tele-label">{t.label}</span>
              <span className="tele-val" style={{ color: t.color }}>
                {cur == null ? "—" : t.fmt(cur)}
              </span>
            </div>
            <Sparkline values={vals} color={t.color} width={92} height={26} />
          </div>
        );
      })}
    </div>
  );
}
