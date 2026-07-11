import type { NetvizDTO } from "../types";
import DuelingSplit from "./DuelingSplit";

const OUT_LABELS = ["L", "S", "R", "B-L", "B-S", "B-R"];

// blue -> cyan -> green -> yellow -> red, by normalized activation intensity.
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

export default function NetworkVisualizer({ netviz }: { netviz: NetvizDTO | null }) {
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
        />
      )}

      <div className="colorbar-row">
        <span className="muted">activation</span>
        <div className="colorbar" />
        <span className="muted">low → high</span>
      </div>

      <Layer label="Input (state)" values={netviz.input} count={s.input_count} />
      <Layer label="Hidden (sampled)" values={netviz.hidden_sample} count={netviz.hidden_count} />

      <div>
        <div className="layer-label"><span>Output (Q)</span><span className="mono">{s.output_count}</span></div>
        <div className="out-cells">
          {netviz.output.map((q, i) => (
            <div
              key={i}
              className={"out-cell" + (i === s.top_index ? " top" : "")}
              style={{ background: heat(Math.abs(q) / maxOut), opacity: i === s.top_index ? 1 : 0.6 }}
              title={q.toFixed(3)}
            >
              {labels[i]}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
