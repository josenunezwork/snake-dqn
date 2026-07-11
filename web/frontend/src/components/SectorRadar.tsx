// A radial "windrose" for the two 16-sector perception features (food density,
// danger map). Each sector is a wedge whose radius/brightness encodes its value.
//
// Geometry mirrors the backend exactly (src/game/snake.py::_angle_to_sector):
//   sector = floor(((atan2(dy, dx) + PI) / 2PI) * n) % n
// with dx,dy in *screen* space (x right, y down). So sector s spans screen angles
// [s/n*2PI - PI, (s+1)/n*2PI - PI). We draw in an SVG whose y also points down, so
// a wedge's on-screen direction is the real direction of that perception — food to
// the snake's right blooms to the right, a wall above blooms upward. "Up on the
// radar = up in the arena."

interface Props {
  values: number[];
  color: [number, number, number];
  label: string;
  // Direction one-hot [Up, Right, Down, Left] — draws the snake's heading arrow.
  heading?: number[] | null;
  // Optional [relX, relY] (screen space) — draws a marker toward a point of interest.
  marker?: [number, number] | null;
  size?: number;
}

const TAU = Math.PI * 2;

// [Up, Right, Down, Left] as screen unit vectors (y points down).
const HEADING_VECS: [number, number][] = [
  [0, -1],
  [1, 0],
  [0, 1],
  [-1, 0],
];

function polar(r: number, a: number): [number, number] {
  return [r * Math.cos(a), r * Math.sin(a)];
}

// Annular wedge path for sector s (inner ri -> outer ro), n sectors total.
function wedgePath(s: number, n: number, ri: number, ro: number): string {
  const a0 = (s / n) * TAU - Math.PI;
  const a1 = ((s + 1) / n) * TAU - Math.PI;
  const [x0o, y0o] = polar(ro, a0);
  const [x1o, y1o] = polar(ro, a1);
  const [x0i, y0i] = polar(ri, a0);
  const [x1i, y1i] = polar(ri, a1);
  // sweep 1 = increasing angle (clockwise on a y-down canvas).
  return (
    `M${x0i.toFixed(2)} ${y0i.toFixed(2)}` +
    `L${x0o.toFixed(2)} ${y0o.toFixed(2)}` +
    `A${ro} ${ro} 0 0 1 ${x1o.toFixed(2)} ${y1o.toFixed(2)}` +
    `L${x1i.toFixed(2)} ${y1i.toFixed(2)}` +
    `A${ri} ${ri} 0 0 0 ${x0i.toFixed(2)} ${y0i.toFixed(2)}Z`
  );
}

export default function SectorRadar({
  values,
  color,
  label,
  heading = null,
  marker = null,
  size = 116,
}: Props) {
  const n = values.length || 1;
  const [r, g, b] = color;
  const R = size / 2;
  const pad = 4;
  const ri = R * 0.16; // inner hole
  const ro = R - pad; // max outer radius
  const rgb = `rgb(${r},${g},${b})`;

  const peak = Math.max(...values.map((v) => (Number.isFinite(v) ? v : 0)), 1e-6);
  const total = values.reduce((acc, v) => acc + (Number.isFinite(v) ? Math.max(0, v) : 0), 0);

  // Heading arrow (from the direction one-hot, if provided).
  let headVec: [number, number] | null = null;
  if (heading && heading.length >= 4) {
    const idx = heading.indexOf(Math.max(...heading.slice(0, 4)));
    if (idx >= 0 && heading[idx] > 0.5) headVec = HEADING_VECS[idx];
  }

  // Point-of-interest marker (e.g. nearest food) along its screen direction.
  let markVec: [number, number] | null = null;
  if (marker && (Math.abs(marker[0]) > 1e-4 || Math.abs(marker[1]) > 1e-4)) {
    const m = Math.hypot(marker[0], marker[1]) || 1;
    markVec = [marker[0] / m, marker[1] / m];
  }

  return (
    <div className="radar" role="img" aria-label={`${label}: ${total.toFixed(1)} across ${n} sectors`}>
      <svg viewBox={`${-R} ${-R} ${size} ${size}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {/* faint sector grid so empty sectors still read as a ring */}
        {values.map((_, s) => (
          <path key={`bg${s}`} d={wedgePath(s, n, ri, ro)} className="radar-bg" />
        ))}
        {/* range rings */}
        <circle cx="0" cy="0" r={ro} className="radar-ring" />
        <circle cx="0" cy="0" r={ri + (ro - ri) * 0.5} className="radar-ring faint" />
        {/* value wedges */}
        {values.map((v, s) => {
          const t = Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
          if (t <= 0.001) return null;
          const rr = ri + t * (ro - ri);
          // brighter + more opaque as the value climbs; also lifts hot sectors
          const op = 0.25 + 0.65 * t;
          const lift = 1 + 0.5 * t;
          const cc = `rgb(${Math.min(255, r * lift) | 0},${Math.min(255, g * lift) | 0},${Math.min(255, b * lift) | 0})`;
          return (
            <path
              key={`v${s}`}
              d={wedgePath(s, n, ri, rr)}
              fill={cc}
              fillOpacity={op}
              stroke={cc}
              strokeOpacity={Math.min(1, op + 0.15)}
              strokeWidth={0.5}
            >
              <title>{`sector ${s}: ${v.toFixed(2)}`}</title>
            </path>
          );
        })}
        {/* nearest-point-of-interest marker on the rim */}
        {markVec && (
          <g>
            <line
              x1={markVec[0] * (ri + 1)}
              y1={markVec[1] * (ri + 1)}
              x2={markVec[0] * ro}
              y2={markVec[1] * ro}
              stroke="#ffe28c"
              strokeWidth={1}
              strokeDasharray="2 2"
              opacity={0.8}
            />
            <circle cx={markVec[0] * ro} cy={markVec[1] * ro} r={2.4} fill="#ffe28c" />
          </g>
        )}
        {/* heading arrow at the hub */}
        {headVec && (
          <g className="radar-head">
            <line x1="0" y1="0" x2={headVec[0] * ri * 1.7} y2={headVec[1] * ri * 1.7} stroke="#e6edf6" strokeWidth={1.6} strokeLinecap="round" />
            <circle cx={headVec[0] * ri * 1.7} cy={headVec[1] * ri * 1.7} r={1.8} fill="#e6edf6" />
          </g>
        )}
        <circle cx="0" cy="0" r={ri * 0.6} className="radar-hub" />
      </svg>
      <div className="radar-cap">
        <span className="radar-label">{label}</span>
        <span className="radar-peak mono" style={{ color: rgb }}>
          peak {peak.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
