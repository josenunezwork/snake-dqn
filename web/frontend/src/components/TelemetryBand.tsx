import { memo, useMemo } from "react";
import type { MetricSample } from "../types";
import Sparkline from "./Sparkline";
import InfoDot from "./InfoDot";

interface Tile {
  key: string;
  label: string;
  color: string;
  term?: string; // glossary key for the ⓘ explainer
  get: (s: MetricSample) => number | null;
  fmt: (v: number) => string;
}

const compact = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(Math.round(v)));

// Series colors route through CSS variables (with the historical dark-theme hex
// as fallback) so the styles stream can supply light-theme variants.
const TILES: Tile[] = [
  { key: "best", label: "Best len", color: "var(--viz-blue, #38bdf8)", term: "bestlen", get: (s) => s.bestLength, fmt: (v) => String(Math.round(v)) },
  { key: "alive", label: "Alive", color: "var(--viz-green, #34d399)", term: "alive", get: (s) => s.alive, fmt: (v) => String(Math.round(v)) },
  { key: "food", label: "Food", color: "var(--viz-amber, #f59e0b)", term: "food", get: (s) => s.foodEaten, fmt: compact },
  { key: "kills", label: "Kills", color: "var(--viz-pink, #f472b6)", term: "kills", get: (s) => s.kills, fmt: (v) => String(Math.round(v)) },
  { key: "loss", label: "Loss", color: "var(--viz-red, #f87171)", term: "loss", get: (s) => s.loss, fmt: (v) => v.toFixed(3) },
  { key: "eps", label: "ε", color: "var(--viz-purple, #a78bfa)", term: "epsilon", get: (s) => s.epsilon, fmt: (v) => v.toFixed(2) },
];

function lastNonNull(vals: (number | null)[]): number | null {
  for (let i = vals.length - 1; i >= 0; i--) if (vals[i] != null) return vals[i];
  return null;
}

interface Props {
  history: MetricSample[];
  // True while the engine is paused (heartbeat frames). The hook owns the ring
  // buffer, but the band also defends itself: consecutive samples with an
  // unadvanced frame counter (paused heartbeats / duplicate frames) are dropped
  // so a paused engine cannot flatline the sparklines with repeats.
  paused?: boolean;
}

// Always-visible "telemetry terminal" strip: a rolling sparkline per key live
// metric. Reads the ring buffer from useGameSocket; renders nothing until data
// arrives so it never flashes empty.
function TelemetryBand({ history, paused = false }: Props) {
  // Drop consecutive duplicate frames (frame counter not advanced).
  const samples = useMemo(() => {
    const out: MetricSample[] = [];
    for (const s of history) {
      const prev = out[out.length - 1];
      if (prev && prev.frame === s.frame) continue;
      out.push(s);
    }
    return out;
  }, [history]);

  if (samples.length < 2) return null;
  return (
    <div className="telemetry" role="group" aria-label="Live telemetry" data-paused={paused || undefined}>
      {TILES.map((t) => {
        const vals = samples.map(t.get);
        const cur = lastNonNull(vals);
        return (
          <div className="tele-tile" key={t.key}>
            <div className="tele-head">
              <span className="tele-label">
                {t.label}
                {t.term && <InfoDot term={t.term} />}
              </span>
              <span className="tele-val" style={{ color: t.color }}>
                {cur == null ? "—" : t.fmt(cur)}
              </span>
            </div>
            <Sparkline values={vals} color={t.color} width={92} height={26} showRange format={t.fmt} />
          </div>
        );
      })}
    </div>
  );
}

export default memo(TelemetryBand);
