import { memo, useEffect, useMemo, useRef, useState } from "react";
import type { HeroRasterDTO } from "../types";

// "Ego-raster viewer" — renders what the raster31v2 snake actually sees: the
// heading-rotated tactical stack, drawn as a 31x31 grid with the snake's head at
// the centre facing UP (the planes arrive already ego-rotated from the backend,
// so no rotation is done here — up on the grid is straight ahead for the snake).
//
// Two views:
//   • Composite — every cell painted by its type code (own body, enemy, food,
//     wall, predicted-next, corpse…) in a distinct colour. The whole world at once.
//   • Single channel — one plane isolated, cell brightness = its value byte
//     (e.g. own_body value = tail-low→head-high TTL; enemy_head = relative size).
//
// Only rendered when the served policy is raster31v2; on vector61 there is no
// raster and the caller shows a graceful placeholder instead.

// Type-code → display colour + human label. Mirrors the codes emitted by
// web/backend/serialize.py::_hero_raster (tactical_codes). Code 0 (empty) is the
// grid background and never painted.
interface CodeStyle {
  code: number;
  key: string;
  label: string;
  color: [number, number, number];
}

const CODE_STYLES: CodeStyle[] = [
  { code: 8, key: "own_head", label: "Own head", color: [56, 189, 248] },
  { code: 5, key: "own_body", label: "Own body", color: [37, 118, 153] },
  { code: 7, key: "enemy_head", label: "Enemy head", color: [248, 113, 113] },
  { code: 6, key: "enemy_body", label: "Enemy body", color: [153, 60, 60] },
  { code: 4, key: "enemy_pred", label: "Predicted next", color: [251, 146, 60] },
  { code: 2, key: "ambient_food", label: "Food", color: [74, 222, 128] },
  { code: 3, key: "corpse_food", label: "Corpse food", color: [163, 230, 53] },
  { code: 1, key: "wall", label: "Wall", color: [100, 116, 139] },
];

const rgb = ([r, g, b]: [number, number, number]) => `rgb(${r},${g},${b})`;
const rgba = ([r, g, b]: [number, number, number], a: number) => `rgba(${r},${g},${b},${a})`;

// Canvas backing size (device-independent CSS px); grid is drawn to fill it.
const CANVAS = 248;

// Strategic density plane colours, keyed by the channel names the backend
// serves (src/simd_env/featurizer.py STRATEGIC_CHANNEL_NAMES).
const STRATEGIC_COLORS: Record<string, [number, number, number]> = {
  enemy_mass_density: [248, 113, 113],
  food_mass_density: [74, 222, 128],
  own_body_density: [56, 189, 248],
};
const STRATEGIC_FALLBACK: [number, number, number] = [148, 163, 184];

// Scalar-band grouping fallback. Mirrors web/backend/serialize.py
// RASTER31V2_SCALAR_GROUPS (the per-index contract of
// src.simd_env.featurizer._build_scalars); used when the payload does not ship
// its own labels (`scalar_groups`).
const SCALAR_GROUPS_FALLBACK: { name: string; start: number; end: number }[] = [
  { name: "Length (norm + log)", start: 0, end: 2 },
  { name: "Boost (available + cost phase)", start: 2, end: 4 },
  { name: "Hunger (frames since food)", start: 4, end: 5 },
  { name: "Episode progress", start: 5, end: 6 },
  { name: "Snakes alive (fraction)", start: 6, end: 7 },
  { name: "Mass rank percentile", start: 7, end: 8 },
  { name: "Wall dist ego (A/R/B/L)", start: 8, end: 12 },
  { name: "Nearest food ego (dx/dy/dist)", start: 12, end: 15 },
  { name: "Nearest enemy (dx/dy/size/boost)", start: 15, end: 19 },
  { name: "2nd enemy (dx/dy/size/boost)", start: 19, end: 23 },
  { name: "World pos (x/y)", start: 23, end: 25 },
  { name: "Arena type", start: 25, end: 26 },
];

// Action labels for the 6-bit legality mask (0 L, 1 S, 2 R, 3-5 boost variants).
const MASK_LABELS = ["L", "S", "R", "B-L", "B-S", "B-R"];

// One 25x25 strategic density plane, painted to a small canvas: brightness =
// density byte, tinted per channel.
function StrategicPlane({
  plane,
  name,
  size,
}: {
  plane: number[][];
  name: string;
  size: number;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const PX = 76;
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const S = size || plane.length || 25;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    canvas.width = PX * dpr;
    canvas.height = PX * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, PX, PX);
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    ctx.fillRect(0, 0, PX, PX);
    const color = STRATEGIC_COLORS[name] ?? STRATEGIC_FALLBACK;
    const cell = PX / S;
    for (let y = 0; y < S; y++) {
      const row = plane[y] || [];
      for (let x = 0; x < S; x++) {
        const v = (row[x] ?? 0) / 255;
        if (v <= 0) continue;
        ctx.fillStyle = rgba(color, 0.15 + 0.85 * v);
        ctx.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
  }, [plane, name, size]);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
      <canvas
        ref={ref}
        className="raster-canvas"
        style={{ width: PX, height: PX }}
        role="img"
        aria-label={`Strategic plane ${prettyChannel(name)}`}
      />
      <span className="muted" style={{ fontSize: 10, textAlign: "center" }}>
        {prettyChannel(name)}
      </span>
    </div>
  );
}

interface Props {
  raster: HeroRasterDTO | null;
  obsSpec?: string;
  labels?: boolean; // View-menu "labels" toggle: annotate the centre/axes
}

function EgoRasterViewer({ raster, obsSpec, labels = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // "composite" or a channel index (0-based into tactical_channels).
  const [mode, setMode] = useState<"composite" | number>("composite");

  const isRaster = obsSpec === "raster31v2";

  // Which type codes actually appear this frame (for a compact, live legend).
  const presentCodes = useMemo(() => {
    if (!raster) return new Set<number>();
    const seen = new Set<number>();
    for (const row of raster.tactical_code) for (const c of row) if (c) seen.add(c);
    return seen;
  }, [raster]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !raster) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const S = raster.tactical_size || raster.tactical_code.length || 31;
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    canvas.width = CANVAS * dpr;
    canvas.height = CANVAS * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, CANVAS, CANVAS);

    const cell = CANVAS / S;
    const code = raster.tactical_code;
    const val = raster.tactical_value;

    // backdrop
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    ctx.fillRect(0, 0, CANVAS, CANVAS);

    const styleByCode = new Map(CODE_STYLES.map((s) => [s.code, s]));

    if (mode === "composite") {
      for (let y = 0; y < S; y++) {
        const crow = code[y] || [];
        for (let x = 0; x < S; x++) {
          const c = crow[x];
          if (!c) continue;
          const st = styleByCode.get(c);
          if (!st) continue;
          // value byte modulates opacity so structure (head vs tail, dense food)
          // reads even in the composite view; floor so cells never vanish.
          const v = (val[y]?.[x] ?? 0) / 255;
          ctx.fillStyle = rgba(st.color, 0.55 + 0.45 * v);
          ctx.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
    } else {
      // single channel: paint that channel's cells by value-byte brightness.
      const chStyle = CODE_STYLES.find(
        (s) => raster.tactical_codes[s.key] !== undefined && channelOf(raster, s.key) === mode
      );
      const color = chStyle ? chStyle.color : ([56, 189, 248] as [number, number, number]);
      const wantCode = codeForChannel(raster, mode);
      for (let y = 0; y < S; y++) {
        const crow = code[y] || [];
        for (let x = 0; x < S; x++) {
          if (crow[x] !== wantCode) continue;
          const v = (val[y]?.[x] ?? 0) / 255;
          ctx.fillStyle = rgba(color, 0.2 + 0.8 * v);
          ctx.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
    }

    // grid lines (subtle) + centre crosshair marking the snake's head.
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let i = 1; i < S; i++) {
      const p = Math.round(i * cell) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, CANVAS);
      ctx.moveTo(0, p);
      ctx.lineTo(CANVAS, p);
      ctx.stroke();
    }
    // centre cell = ego origin (snake head). Ring it so orientation is obvious.
    const mid = Math.floor(S / 2);
    ctx.strokeStyle = "rgba(230,237,246,0.8)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(mid * cell + 1, mid * cell + 1, cell - 2, cell - 2);
    // "facing up" arrow above the head cell.
    ctx.fillStyle = "rgba(230,237,246,0.85)";
    const cx = mid * cell + cell / 2;
    const ay = mid * cell - cell * 0.4;
    ctx.beginPath();
    ctx.moveTo(cx, ay - cell * 0.8);
    ctx.lineTo(cx - cell * 0.45, ay);
    ctx.lineTo(cx + cell * 0.45, ay);
    ctx.closePath();
    ctx.fill();
  }, [raster, mode]);

  if (!isRaster) {
    return (
      <div className="panel">
        <div className="empty-state">
          <div className="empty-glyph">▦</div>
          <div className="empty-title">No ego-raster for this model</div>
          <div className="muted" style={{ fontSize: 12 }}>
            The served policy is <span className="mono">{obsSpec ?? "vector61"}</span> — a
            hand-crafted feature vector, not a spatial raster. Load a{" "}
            <span className="mono">raster31v2</span> checkpoint (Controls → Model — raster
            training outputs appear there as <span className="mono">runs/…/latest_pqn.pth</span>)
            to watch what the snake sees.
          </div>
        </div>
      </div>
    );
  }

  if (!raster) {
    return (
      <div className="panel">
        <div className="empty-state">
          <div className="empty-glyph">▦</div>
          <div className="empty-title">No observation yet</div>
          <div className="muted" style={{ fontSize: 12 }}>
            Waiting for the hero snake to produce a raster observation.
          </div>
        </div>
      </div>
    );
  }

  const S = raster.tactical_size;
  const channels = raster.tactical_channels;

  return (
    <div className="panel">
      <div className="panel-intro">
        The snake's eye-view — the tactical raster it actually feeds its network, rotated so the
        head faces up.
      </div>

      <div className="card">
        <div className="section-title">
          Ego-raster · {S}×{S} tactical
        </div>

        <div className="raster-modes" role="tablist" aria-label="Raster channel">
          <button
            role="tab"
            aria-selected={mode === "composite"}
            className={"raster-chip" + (mode === "composite" ? " on" : "")}
            onClick={() => setMode("composite")}
          >
            Composite
          </button>
          {channels.map((name, i) => {
            // only offer channels that map to a paintable type code
            if (codeForChannel(raster, i) === null) return null;
            return (
              <button
                key={name}
                role="tab"
                aria-selected={mode === i}
                className={"raster-chip" + (mode === i ? " on" : "")}
                onClick={() => setMode(i)}
              >
                {prettyChannel(name)}
              </button>
            );
          })}
        </div>

        <div className="raster-stage">
          <canvas
            ref={canvasRef}
            className="raster-canvas"
            style={{ width: CANVAS, height: CANVAS }}
            role="img"
            aria-label={
              mode === "composite"
                ? `Ego raster composite, ${S} by ${S}`
                : `Ego raster channel ${prettyChannel(channels[mode as number])}`
            }
          />
          {labels && (
            <div className="raster-axis muted mono" aria-hidden="true">
              ▲ ahead
            </div>
          )}
        </div>

        <div className="raster-legend">
          {mode === "composite"
            ? CODE_STYLES.map((s) => (
                <div
                  key={s.key}
                  className={"raster-legend-item" + (presentCodes.has(s.code) ? "" : " off")}
                >
                  <span className="raster-swatch" style={{ background: rgb(s.color) }} />
                  <span>{s.label}</span>
                </div>
              ))
            : (() => {
                const st = CODE_STYLES.find((s) => codeForChannel(raster, mode as number) === s.code);
                return (
                  <div className="raster-legend-item">
                    <span
                      className="raster-swatch"
                      style={{
                        background: st ? rgb(st.color) : "var(--accent)",
                      }}
                    />
                    <span>
                      {st?.label ?? prettyChannel(channels[mode as number])} — brightness = value
                    </span>
                  </div>
                );
              })()}
        </div>

        <div className="muted" style={{ fontSize: 11, marginTop: 8, lineHeight: 1.6 }}>
          Centre cell (ringed) is the snake's head; the arrow is straight ahead. The stack is
          heading-rotated, so a wall on the left of the grid is a wall to the snake's left
          regardless of which way it points in the arena.
        </div>
      </div>

      {raster.mask && raster.mask.length > 0 && (
        <div className="card">
          <div className="section-title">Action mask · which moves are legal</div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }} role="group" aria-label="Legal actions">
            {raster.mask.map((legal, i) => (
              <span
                key={i}
                className="mono"
                aria-label={`${MASK_LABELS[i] ?? i}: ${legal ? "legal" : "masked"}`}
                style={{
                  flex: "1 0 auto",
                  textAlign: "center",
                  fontSize: 11,
                  padding: "4px 8px",
                  borderRadius: 6,
                  border: "1px solid var(--border)",
                  color: legal ? "var(--text)" : "var(--muted)",
                  opacity: legal ? 1 : 0.45,
                  textDecoration: legal ? "none" : "line-through",
                }}
              >
                {MASK_LABELS[i] ?? i}
              </span>
            ))}
          </div>
          <div className="muted" style={{ fontSize: 10, marginTop: 6 }}>
            Struck-out actions are masked before the argmax — the network cannot pick them.
          </div>
        </div>
      )}

      {raster.strategic && raster.strategic.length > 0 && (
        <div className="card">
          <div className="section-title">
            Strategic · {raster.strategic_size}×{raster.strategic_size} density planes
          </div>
          <div style={{ display: "flex", gap: 10, justifyContent: "space-between" }}>
            {raster.strategic.map((plane, i) => (
              <StrategicPlane
                key={raster.strategic_channels[i] ?? i}
                plane={plane}
                name={raster.strategic_channels[i] ?? `plane_${i}`}
                size={raster.strategic_size}
              />
            ))}
          </div>
          <div className="muted" style={{ fontSize: 10, marginTop: 6 }}>
            Coarser, wider view than the tactical grid — mass density per region, also
            heading-rotated. Brightness = density.
          </div>
        </div>
      )}

      {raster.scalars && raster.scalars.length > 0 && (
        <details className="state-details">
          <summary>
            <span className="section-title" style={{ margin: 0, display: "inline" }}>
              Scalars ({raster.scalars.length}-D)
            </span>
          </summary>
          {scalarGroupsFor(raster).map((grp) => (
            <div className="card" key={grp.name} style={{ padding: "8px 12px" }}>
              <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>
                {grp.name}{" "}
                <span className="mono">
                  [{grp.start}–{grp.end - 1}]
                </span>
              </div>
              <div className="mono" style={{ fontSize: 12 }}>
                {raster.scalars
                  .slice(grp.start, grp.end)
                  .map((v) => v.toFixed(2))
                  .join("  ")}
              </div>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}

// ---- helpers -------------------------------------------------------------

// The tactical channel index a given code expands into. tactical_codes maps
// name -> code; tactical_channels lists names by plane index. We invert.
function channelOf(raster: HeroRasterDTO, key: string): number | null {
  const i = raster.tactical_channels.indexOf(key);
  return i >= 0 ? i : null;
}

// For a channel plane index, the type code whose cells belong to it (or null if
// the plane has no direct code, e.g. the reserved ablation slot).
function codeForChannel(raster: HeroRasterDTO, channel: number): number | null {
  const name = raster.tactical_channels[channel];
  if (name === undefined) return null;
  const code = raster.tactical_codes[name];
  return code === undefined ? null : code;
}

function prettyChannel(name: string): string {
  return name
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// Scalar-band labels: prefer names served in the payload (state_labels-style
// group dicts under `scalar_groups`, when the backend ships them), falling back
// to the local mirror of serialize.py's RASTER31V2_SCALAR_GROUPS.
function scalarGroupsFor(raster: HeroRasterDTO): { name: string; start: number; end: number }[] {
  const served = raster.scalar_groups;
  const groups = served && served.length > 0 ? served : SCALAR_GROUPS_FALLBACK;
  return groups.filter((g) => g.end <= raster.scalars.length);
}

export default memo(EgoRasterViewer);
