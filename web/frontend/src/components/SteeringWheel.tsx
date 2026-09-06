// The 6 Q-values ARE a steering decision: turn Left / go Straight / turn Right,
// each at normal or boost speed. Horizontal bars hide that structure. This dial
// lays them out spatially — three directions fanned around "up", an inner ring
// for normal speed and an outer ring for boost — so "it wants to turn right, and
// would boost" reads at a glance. Intensity encodes each action's relative value;
// the chosen action glows; the hub ring shows decision confidence (the margin).
//
// Action index layout (see src/game action space): 0 L, 1 S, 2 R (normal),
// 3 boost-L, 4 boost-S, 5 boost-R.

interface Props {
  q: number[];
  // The action the agent actually took ("action taken"): the executed action
  // when the backend serves one, else the greedy argmax.
  chosen: number;
  // The greedy argmax, when it differs from `chosen` (exploration/masking).
  // The greedy wedge gets a dashed outline and its own caption line.
  greedy?: number | null;
  labels: string[];
  size?: number;
}

const UP = -Math.PI / 2;
const SPREAD = 0.74; // ~42° between adjacent directions
const HALF = 0.32; // wedge half-width (~18°)

// direction index 0/1/2 -> centre angle (screen space, up = -PI/2)
const DIR_ANGLE = [UP - SPREAD, UP, UP + SPREAD];
const DIR_KEY = ["L", "S", "R"];

function polar(r: number, a: number): [number, number] {
  return [r * Math.cos(a), r * Math.sin(a)];
}

function wedge(a0: number, a1: number, ri: number, ro: number): string {
  const [x0i, y0i] = polar(ri, a0);
  const [x0o, y0o] = polar(ro, a0);
  const [x1o, y1o] = polar(ro, a1);
  const [x1i, y1i] = polar(ri, a1);
  return (
    `M${x0i.toFixed(2)} ${y0i.toFixed(2)}` +
    `L${x0o.toFixed(2)} ${y0o.toFixed(2)}` +
    `A${ro} ${ro} 0 0 1 ${x1o.toFixed(2)} ${y1o.toFixed(2)}` +
    `L${x1i.toFixed(2)} ${y1i.toFixed(2)}` +
    `A${ri} ${ri} 0 0 0 ${x0i.toFixed(2)} ${y0i.toFixed(2)}Z`
  );
}

export function confidence(margin: number): { label: string; color: string } {
  // CSS variables (with dark-theme fallbacks) so the light theme keeps contrast.
  if (margin > 1) return { label: "confident", color: "var(--viz-green, #34d399)" };
  if (margin > 0.3) return { label: "moderate", color: "var(--viz-amber, #fbbf24)" };
  return { label: "close call", color: "var(--viz-red, #f87171)" };
}

export default function SteeringWheel({ q, chosen, greedy = null, labels, size = 172 }: Props) {
  const greedyDiffers = greedy != null && greedy !== chosen;
  const R = size / 2;
  const ri = R * 0.2;
  const rm = R * 0.56; // normal ring outer / boost ring inner
  const ro = R * 0.94;

  const min = Math.min(...q);
  const max = Math.max(...q);
  const range = max - min || 1;
  // second-best gap = decision margin (confidence)
  const sorted = [...q].sort((a, b) => b - a);
  const margin = sorted.length > 1 ? sorted[0] - sorted[1] : 0;
  const conf = confidence(margin);

  // action a -> (direction 0..2, ring 0=normal|1=boost)
  const cells = q.map((val, a) => {
    const dir = a % 3;
    const boost = a >= 3 ? 1 : 0;
    const a0 = DIR_ANGLE[dir] - HALF;
    const a1 = DIR_ANGLE[dir] + HALF;
    const band0 = boost ? rm + 1 : ri;
    const band1 = boost ? ro : rm - 1;
    const t = (val - min) / range; // 0 worst .. 1 best
    return { a, dir, boost, path: wedge(a0, a1, band0, band1), t, val };
  });

  return (
    <div className="wheel">
      <svg viewBox={`${-R} ${-R} ${size} ${size}`} width="100%" preserveAspectRatio="xMidYMid meet">
        <defs>
          <filter id="wheel-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.4" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {cells.map((c) => {
          const isChosen = c.a === chosen;
          const isGreedy = greedyDiffers && c.a === greedy;
          // teal ramp, brighter with value; the action taken is green + glow;
          // when exploration overrode the greedy pick, that wedge is dash-outlined.
          const base = isChosen ? [52, 211, 153] : [56, 189, 248];
          const lift = 0.45 + 0.55 * c.t;
          const fill = `rgb(${(base[0] * lift) | 0},${(base[1] * lift) | 0},${(base[2] * lift) | 0})`;
          return (
            <path
              key={c.a}
              d={c.path}
              fill={fill}
              fillOpacity={isChosen ? 0.95 : 0.22 + 0.5 * c.t}
              stroke={isChosen ? "#5ff0c0" : isGreedy ? "#38bdf8" : "rgba(120,150,190,0.25)"}
              strokeWidth={isChosen ? 1.4 : isGreedy ? 1.2 : 0.5}
              strokeDasharray={isGreedy ? "3 2" : undefined}
              filter={isChosen ? "url(#wheel-glow)" : undefined}
            >
              <title>
                {`${labels[c.a]}: ${c.val.toFixed(2)}` +
                  (greedyDiffers && isChosen ? " (action taken)" : isGreedy ? " (greedy pick)" : "")}
              </title>
            </path>
          );
        })}
        {/* direction ticks just outside the dial */}
        {DIR_ANGLE.map((ang, i) => {
          const [x, y] = polar(ro + 7, ang);
          return (
            <text key={i} x={x} y={y} className="wheel-tick" textAnchor="middle" dominantBaseline="middle">
              {DIR_KEY[i]}
            </text>
          );
        })}
        {/* hub confidence ring (style, not attributes, so CSS vars resolve) */}
        <circle cx="0" cy="0" r={ri * 0.82} fill="none" style={{ stroke: conf.color }} strokeWidth="2.5" opacity="0.9" />
        <circle cx="0" cy="0" r={ri * 0.5} style={{ fill: conf.color }} opacity="0.85" />
      </svg>
      <div className="wheel-cap">
        <span className="wheel-chosen">
          ▸ {labels[chosen] ?? "—"}
          {greedyDiffers && <span className="muted" style={{ fontWeight: 400 }}> · taken</span>}
        </span>
        {greedyDiffers && (
          <span className="wheel-greedy muted" style={{ fontSize: 11 }}>
            greedy pick: {labels[greedy as number] ?? "—"}
          </span>
        )}
        <span className="wheel-conf" style={{ color: conf.color }}>
          {conf.label} · margin {margin >= 0 ? "+" : ""}
          {margin.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
