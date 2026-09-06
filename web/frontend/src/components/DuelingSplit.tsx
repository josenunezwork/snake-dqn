// The network is a Dueling DQN: it estimates a single state-value V(s) — "how
// good is this position, regardless of which move I make" — plus a per-action
// advantage A(s,a). Q = V + (A − mean A). The fused Q hides this; splitting it
// out is the canonical interpretability story for this architecture. Only the
// mean-centered advantage is identifiable, so that's what we plot.

import InfoDot from "./InfoDot";

interface Props {
  value: number;
  advantages: number[];
  labels: string[];
  // The greedy argmax (highest-Q action).
  chosen: number;
  // The action actually executed last step, when the backend serves it and it
  // differs from the greedy pick — that row is marked "taken" and the greedy
  // row keeps its ▸ with a "greedy" tag.
  executed?: number | null;
}

export default function DuelingSplit({ value, advantages, labels, chosen, executed = null }: Props) {
  const mean = advantages.reduce((a, b) => a + b, 0) / (advantages.length || 1);
  const centered = advantages.map((a) => a - mean);
  const maxAbs = Math.max(...centered.map(Math.abs), 1e-6);
  const executedDiffers = executed != null && executed !== chosen;

  return (
    <div className="card dueling">
      <div className="section-title">
        Dueling split · value + advantage
        <InfoDot term="dueling" />
      </div>
      <div className="dueling-value">
        <div className="dv-num mono">{value.toFixed(2)}</div>
        <div className="dv-cap muted">
          V(s) — how good this position is,
          <br />
          regardless of the move.
        </div>
      </div>
      <div className="dueling-adv">
        {centered.map((a, i) => {
          const w = (Math.abs(a) / maxAbs) * 50;
          const left = a >= 0 ? 50 : 50 - w;
          const pos = a >= 0;
          return (
            <div
              className={"adv-row" + (i === (executedDiffers ? executed : chosen) ? " chosen" : "")}
              key={i}
            >
              <span className="adv-label">
                {i === (executedDiffers ? executed : chosen) ? "▸ " : ""}
                {labels[i]}
                {executedDiffers && i === executed && (
                  <span className="muted" style={{ fontSize: 9 }}> taken</span>
                )}
                {executedDiffers && i === chosen && (
                  <span className="muted" style={{ fontSize: 9 }}> greedy</span>
                )}
              </span>
              <div className="adv-track">
                <span className="adv-zero" />
                <span
                  className={"adv-fill " + (pos ? "pos" : "neg")}
                  style={{ left: `${left}%`, width: `${w}%` }}
                />
              </div>
              <span className="adv-val mono">
                {a >= 0 ? "+" : ""}
                {a.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="muted dueling-eq mono">Q(s,a) = V(s) + (A(s,a) − mean A)</div>
    </div>
  );
}
