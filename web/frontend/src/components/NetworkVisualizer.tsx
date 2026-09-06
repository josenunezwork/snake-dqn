import { memo } from "react";
import type { NetvizDTO } from "../types";
import DuelingSplit from "./DuelingSplit";

const OUT_LABELS = ["L", "S", "R", "B-L", "B-S", "B-R"];

// blue -> cyan -> green -> yellow -> red, by normalized activation intensity.
// Used for the input/hidden activation bands (magnitude is the honest story
// there). The output Q row uses the signed diverging scale below instead.
function heat(t: number): string {
  const stops: [number, [number, number, number]][] = [
    [0.0, [30, 60, 120]],
    [0.25, [0, 180, 200]],
    [0.5, [50, 205, 50]],
    [0.75, [255, 200, 0]],
    [1.0, [240, 80, 60]],
  ];
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (x <= stops[i][0]) {
      const [a, ca] = stops[i - 1];
      const [b, cb] = stops[i];
      const f = (x - a) / (b - a || 1);
      const c = ca.map((v, k) => Math.round(v + (cb[k] - v) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(240,80,60)";
}

type RGB = [number, number, number];

const Q_COOL: RGB = [37, 99, 235]; // worst (most negative Q) — cool blue
const Q_MID: RGB = [71, 85, 105]; // neutral slate at Q ≈ 0
const Q_WARM: RGB = [245, 158, 11]; // best (most positive Q) — warm amber

function mix(a: RGB, b: RGB, f: number): RGB {
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

// Signed diverging scale for the output Q row: negative Q → cool, positive Q →
// warm, centered at zero. |Q| alone conflated "strongly want" with "strongly
// avoid" (the most negative Q rendered hottest).
function qColor(q: number, maxAbs: number): RGB {
  const t = Math.max(-1, Math.min(1, q / (maxAbs || 1)));
  return t >= 0 ? mix(Q_MID, Q_WARM, t) : mix(Q_MID, Q_COOL, -t);
}

// Pick the label color by background luminance so the action letters stay
// readable on both ends of the ramp (near-black on light cells, near-white on
// dark cells).
function labelColorFor([r, g, b]: RGB): string {
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return lum > 140 ? "#07111a" : "#f2f6fb";
}

const rgbStr = ([r, g, b]: RGB) => `rgb(${r},${g},${b})`;

function Layer({ label, values, count }: { label: string; values: number[]; count: number }) {
  const max = Math.max(...values.map(Math.abs), 1e-6);
  return (
    <div>
      <div className="layer-label">
        <span>{label}</span>
        <span className="mono">{count}</span>
      </div>
      <div className="cells">
        {values.map((v, i) => (
          <div key={i} className="cell" style={{ background: heat(Math.abs(v) / max) }} title={v.toFixed(3)} />
        ))}
      </div>
    </div>
  );
}

interface Props {
  netviz: NetvizDTO | null;
  // Served obs contract; drives the honest input-band label for raster models.
  obsSpec?: string;
  // The action the hero actually executed last step (frame.inspector
  // executed_action), when it differs from the argmax; forwarded to the
  // dueling split.
  executedAction?: number | null;
}

function NetworkVisualizer({ netviz, obsSpec, executedAction = null }: Props) {
  if (!netviz || netviz.summary.status !== "READY") {
    return (
      <div className="panel">
        <div className="empty-state">
          <div className="empty-glyph">⌁</div>
          <div className="empty-title">Waiting for a live forward pass</div>
          <div className="muted" style={{ fontSize: 12 }}>
            The network lights up once the inspected snake is alive and deciding.
          </div>
        </div>
      </div>
    );
  }
  const s = netviz.summary;
  const maxOut = Math.max(...netviz.output.map(Math.abs), 1e-6);
  const labels = OUT_LABELS;
  const topIdx = s.top_index;
  // Honest input-band label: for raster models the "input" band is only the
  // 26-D scalar slice of a much larger raster observation. Infer from the band
  // length when obs_spec has not been wired through by the caller.
  const isRaster = obsSpec === "raster31v2" || (obsSpec === undefined && netviz.input.length === 26);
  const inputLabel = isRaster ? "Scalar band — part of the raster input" : "Input (state)";

  return (
    <div className="panel netviz">
      <div className="panel-intro">A live look inside the network as it decides — brighter cells are more active.</div>
      <div className="card">
        <div className="section-title">{s.architecture}</div>
        <div className="statgrid">
          <div className="stat"><div className="k">Top action</div><div className="v">{s.top_action}</div></div>
          <div className="stat"><div className="k">Top Q</div><div className="v">{s.top_q.toFixed(2)}</div></div>
          <div className="stat"><div className="k">Margin</div><div className="v">{s.margin.toFixed(2)}</div></div>
          <div className="stat"><div className="k">Hidden act</div><div className="v">{s.hidden_activity.toFixed(2)}</div></div>
        </div>
      </div>

      {netviz.value != null && netviz.advantages && netviz.advantages.length > 0 && (
        <DuelingSplit
          value={netviz.value}
          advantages={netviz.advantages}
          labels={labels}
          chosen={topIdx}
          executed={executedAction}
        />
      )}

      <div className="colorbar-row">
        <span className="muted">activation</span>
        <div className="colorbar" />
        <span className="muted">low → high</span>
      </div>

      <Layer label={inputLabel} values={netviz.input} count={s.input_count} />
      <Layer label="Hidden (sampled)" values={netviz.hidden_sample} count={netviz.hidden_count} />

      <div>
        <div className="layer-label"><span>Output (Q)</span><span className="mono">{s.output_count}</span></div>
        <div className="out-cells">
          {netviz.output.map((q, i) => {
            const isTop = i === s.top_index;
            let bg = qColor(q, maxOut);
            // Dim non-argmax cells by blending the BACKGROUND toward neutral —
            // never by fading the whole cell, which took the label with it.
            if (!isTop) bg = mix(bg, [40, 50, 66], 0.35);
            return (
              <div
                key={i}
                className={"out-cell" + (isTop ? " top" : "")}
                style={{ background: rgbStr(bg), color: labelColorFor(bg) }}
                title={`${labels[i] ?? i}: ${q.toFixed(3)}${isTop ? " (greedy pick)" : ""}`}
              >
                {labels[i]}
              </div>
            );
          })}
        </div>
        <div className="muted" style={{ fontSize: 10, marginTop: 4 }}>
          Q colors are signed: cool = worst, warm = best. The outlined cell is the greedy pick.
        </div>
      </div>
    </div>
  );
}

export default memo(NetworkVisualizer);
