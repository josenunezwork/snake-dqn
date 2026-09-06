import { memo, type CSSProperties } from "react";

interface Props {
  values: (number | null)[];
  width?: number;
  height?: number;
  color?: string;
  fill?: boolean;
  strokeWidth?: number;
  // Annotate the window min/max as tiny tabular-nums labels so the auto-scaled
  // line cannot mislead about trend magnitude (a 0.101→0.103 jitter fills the
  // full height either way — the labels say so).
  showRange?: boolean;
  // Formatter for the range labels (defaults to a compact numeric format).
  format?: (v: number) => string;
}

const compactFmt = (v: number): string => {
  const a = Math.abs(v);
  if (a >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (a >= 10 || Number.isInteger(v)) return String(Math.round(v));
  return v.toFixed(2);
};

// Tiny dependency-free SVG sparkline. Auto-scales to its own min/max over the
// window and breaks the line across null gaps (e.g. intermittent training loss)
// instead of plotting them as zero. Colors may be CSS variables (e.g.
// "var(--viz-amber, #fbbf24)") — they are applied via style, not attributes,
// so var() resolves.
function Sparkline({
  values,
  width = 68,
  height = 26,
  color = "#38bdf8",
  fill = true,
  strokeWidth = 1.5,
  showRange = false,
  format = compactFmt,
}: Props) {
  const present = values.filter((v): v is number => v != null);
  if (present.length < 2) {
    return <svg width={width} height={height} className="spark" aria-hidden="true" />;
  }
  let min = Infinity;
  let max = -Infinity;
  for (const v of present) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const n = values.length;
  const pad = strokeWidth;
  const px = (i: number) => (n === 1 ? width / 2 : (i / (n - 1)) * (width - pad * 2) + pad);
  const py = (v: number) => height - pad - ((v - min) / range) * (height - pad * 2);

  // Line path, broken across gaps.
  let line = "";
  let started = false;
  values.forEach((v, i) => {
    if (v == null) {
      started = false;
      return;
    }
    line += `${started ? "L" : "M"}${px(i).toFixed(1)} ${py(v).toFixed(1)} `;
    started = true;
  });

  const hasGaps = present.length !== n;
  // Area fill only when contiguous (a gapped area polygon would be misleading).
  let area = "";
  if (fill && !hasGaps) {
    area =
      `M${px(0).toFixed(1)} ${py(values[0] as number).toFixed(1)} ` +
      values
        .map((v, i) => `L${px(i).toFixed(1)} ${py(v as number).toFixed(1)}`)
        .join(" ") +
      ` L${px(n - 1).toFixed(1)} ${height} L${px(0).toFixed(1)} ${height} Z`;
  }
  const gid = `spk-${color.replace(/[^a-z0-9]/gi, "")}`;

  const rangeLabelStyle: CSSProperties = {
    fill: "var(--muted, #8a97a8)",
    fontFamily: "var(--mono, monospace)",
    fontSize: 6.5,
    fontVariantNumeric: "tabular-nums",
  };

  return (
    <svg width={width} height={height} className="spark" viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      {area && (
        <>
          <defs>
            <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" style={{ stopColor: color }} stopOpacity="0.35" />
              <stop offset="100%" style={{ stopColor: color }} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={area} fill={`url(#${gid})`} stroke="none" />
        </>
      )}
      <path
        d={line}
        fill="none"
        style={{ stroke: color }}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {showRange && (
        <g className="spark-range" style={rangeLabelStyle}>
          <text x={width - 1} y={7} textAnchor="end">
            {format(max)}
          </text>
          {max !== min && (
            <text x={width - 1} y={height - 1.5} textAnchor="end">
              {format(min)}
            </text>
          )}
        </g>
      )}
    </svg>
  );
}

export default memo(Sparkline);
