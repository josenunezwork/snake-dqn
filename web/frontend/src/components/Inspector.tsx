import type { InspectorDTO, StateGroup } from "../types";
import SectorRadar from "./SectorRadar";
import SteeringWheel from "./SteeringWheel";
import DecisionTrace from "./DecisionTrace";
import InfoDot from "./InfoDot";

function valueColor(v: number): string {
  // diverging: negative -> red, ~0 -> grey, positive -> teal
  if (v >= 0) return `rgba(56,189,248,${Math.min(0.25 + Math.abs(v), 1)})`;
  return `rgba(248,113,113,${Math.min(0.25 + Math.abs(v), 1)})`;
}

function FeatureRow({ idx, value }: { idx: number; value: number }) {
  const w = Math.min(Math.abs(value), 1) * 100;
  const left = value >= 0 ? 50 : 50 - w / 2;
  return (
    <div className="feat-row">
      <span className="idx">{idx}</span>
      <div className="feat-bar">
        <div
          className="feat-fill"
          style={{ left: `${left}%`, width: `${w / 2}%`, background: valueColor(value) }}
        />
      </div>
      <span className="feat-val">{value.toFixed(2)}</span>
    </div>
  );
}

// A 16-wide state group rendered as a radial windrose instead of 16 flat bars.
function isSectorGroup(g: StateGroup): boolean {
  return g.end - g.start === 16;
}

interface Props {
  inspector: InspectorDTO | null;
  tick?: number;
  heroId?: number;
}

export default function Inspector({ inspector, tick = 0, heroId = 0 }: Props) {
  if (!inspector) {
    return (
      <div className="panel">
        <div className="empty-state">
          <div className="empty-glyph">◎</div>
          <div className="empty-title">No snake to inspect</div>
          <div className="muted" style={{ fontSize: 12 }}>
            The inspected snake isn't alive. Click a live snake in the arena, or pick one in
            Controls, to watch its decision.
          </div>
        </div>
      </div>
    );
  }
  const { state, groups, q_values, action_labels, chosen, free_space } = inspector;
  const sorted = [...q_values].sort((a, b) => b - a);
  const margin = sorted.length > 1 ? sorted[0] - sorted[1] : 0;

  const foodGroup = groups.find((g) => isSectorGroup(g) && g.name.toLowerCase().includes("food"));
  const dangerGroup = groups.find((g) => isSectorGroup(g) && g.name.toLowerCase().includes("danger"));
  const heading = state.slice(0, 4);
  const foodMarker: [number, number] | null = state.length > 6 ? [state[5], state[6]] : null;

  return (
    <div className="panel">
      <div className="panel-intro">The AI's decision right now — what it sees and how it moves.</div>

      <div className="card">
        <div className="section-title">
          Steering · Q-values → action
          <InfoDot term="qvalue" />
          <InfoDot term="margin" />
        </div>
        <div className="decision">
          <SteeringWheel q={q_values} chosen={chosen} labels={action_labels} />
          <div className="qgrid">
            {q_values.map((q, i) => (
              <div key={i} className={"qcell" + (i === chosen ? " chosen" : "")}>
                <span className="qcell-k">
                  {i === chosen ? "▸ " : ""}
                  {action_labels[i]}
                </span>
                <span className="qcell-v mono">{q.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
        <DecisionTrace
          tick={tick}
          chosen={chosen}
          margin={margin}
          labels={action_labels}
          resetKey={`${heroId}:${inspector.input_size}`}
        />
      </div>

      {free_space && (
        <div className="card">
          <div className="section-title">
            Free-space (don't trap yourself)
            <InfoDot term="freespace" />
          </div>
          <div className="row" style={{ marginBottom: 0 }}>
            {["Left", "Straight", "Right"].map((lbl, i) => (
              <div key={lbl} style={{ flex: 1, textAlign: "center" }}>
                <div className="muted" style={{ fontSize: 11 }}>{lbl}</div>
                <div className="mono" style={{ fontSize: 18 }}>{free_space[i].toFixed(2)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(foodGroup || dangerGroup) && (
        <div className="card">
          <div className="section-title">
            Perception · what the agent senses around it
            <InfoDot term="perception" />
          </div>
          <div className="radar-row">
            {foodGroup && (
              <SectorRadar
                label="Food"
                color={[245, 158, 11]}
                values={state.slice(foodGroup.start, foodGroup.end)}
                heading={heading}
                marker={foodMarker}
              />
            )}
            {dangerGroup && (
              <SectorRadar
                label="Danger"
                color={[248, 113, 113]}
                values={state.slice(dangerGroup.start, dangerGroup.end)}
                heading={heading}
              />
            )}
          </div>
          <div className="muted" style={{ fontSize: 11, marginTop: 8, lineHeight: 1.6 }}>
            Screen-oriented — up on the dial is up in the arena. The pale arrow is the snake's
            heading{foodGroup ? "; the dashed line points at the nearest food" : ""}.
          </div>
        </div>
      )}

      <details className="state-details">
        <summary>
          <span className="section-title" style={{ margin: 0, display: "inline" }}>
            Full state vector ({state.length}-D)
          </span>
        </summary>
        <div className="diverging-legend" style={{ marginTop: 10 }}>
          <span className="muted mono">−1</span>
          <div className="legend-bar" />
          <span className="muted mono">0</span>
          <div className="legend-bar pos" />
          <span className="muted mono">+1</span>
        </div>
        {groups.map((grp) => (
          <div className="card" key={grp.name} style={{ padding: "8px 12px" }}>
            <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>
              {grp.name} <span className="mono">[{grp.start}–{grp.end - 1}]</span>
            </div>
            {state.slice(grp.start, grp.end).map((v, j) => (
              <FeatureRow key={grp.start + j} idx={grp.start + j} value={v} />
            ))}
          </div>
        ))}
      </details>
    </div>
  );
}
