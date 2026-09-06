import { useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import type { Frame, SnakeDTO } from "../types";
import DecisionNarrator from "./DecisionNarrator";
import ArenaLegend from "./ArenaLegend";
import EpisodeSummary from "./EpisodeSummary";
import type { ViewSettings } from "../hooks/useViewSettings";

const DEFAULT_VIEW: ViewSettings = {
  accent: "#38bdf8",
  theme: "dark",
  grid: true,
  labels: false,
  crown: true,
  trails: true,
  vignette: true,
};

// Pure renderer: paints whatever the server streamed, but on a local rAF loop so
// motion is smooth (interpolated between the ~12fps server frames) and effects
// (food pulse, particle bursts, boost trails, screen-shake) animate independently
// of the network cadence. It never computes game state — it only tweens between
// the last two frames and reacts to their diff. Scoring stays server-authoritative.

type Pt = [number, number];
interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  born: number;
  life: number;
  size: number;
  color: [number, number, number];
}
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const foodKey = (x: number, y: number) => x * 100000 + y;

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    !!window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

// Lift very dark server colors so a low-value RGB snake is never invisible on the
// dark board, then return the head-bright / tail-dim variants used by the ribbon.
function bodyColors([r, g, b]: [number, number, number]) {
  const lum = 0.299 * r + 0.587 * g + 0.114 * b;
  const FLOOR = 72;
  const k = lum > 0 && lum < FLOOR ? FLOOR / lum : 1;
  const R = Math.min(255, r * k);
  const G = Math.min(255, g * k);
  const B = Math.min(255, b * k);
  return {
    bright: `rgb(${Math.min(R + 55, 255) | 0},${Math.min(G + 55, 255) | 0},${Math.min(B + 55, 255) | 0})`,
    dim: `rgb(${(R * 0.5) | 0},${(G * 0.5) | 0},${(B * 0.5) | 0})`,
    lifted: [R, G, B] as [number, number, number],
  };
}

// Pre-render a soft radial food glow once so we can drawImage() hundreds of
// pellets cheaply (per-pellet radial gradients would thrash the GPU).
function makeFoodSprite(seg: number): HTMLCanvasElement {
  const r = seg * 2.2;
  const c = document.createElement("canvas");
  c.width = c.height = Math.ceil(r * 2);
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(r, r, 0, r, r, r);
  grad.addColorStop(0, "rgba(255,226,140,1)");
  grad.addColorStop(0.28, "rgba(245,158,11,0.95)");
  grad.addColorStop(0.5, "rgba(245,158,11,0.35)");
  grad.addColorStop(1, "rgba(245,158,11,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, c.width, c.height);
  return c;
}

// Cache the static backdrop (board fill + grid + vignette) so we blit it each
// frame instead of re-stroking hundreds of grid lines every rAF tick. Rendered
// at dpr resolution (drawing in arena coords) so the grid stays crisp on HiDPI.
function makeBackdrop(
  W: number,
  H: number,
  seg: number,
  dpr = 1,
  grid = true,
  vignette = true
): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = Math.round(W * dpr);
  c.height = Math.round(H * dpr);
  const ctx = c.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#080b11";
  ctx.fillRect(0, 0, W, H);
  if (grid) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(120,150,190,0.045)";
    ctx.beginPath();
    const step = seg * 4;
    for (let x = step; x < W; x += step) {
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, H);
    }
    for (let y = step; y < H; y += step) {
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(W, y + 0.5);
    }
    ctx.stroke();
  }
  if (vignette) {
    const vg = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.25, W / 2, H / 2, Math.max(W, H) * 0.62);
    vg.addColorStop(0, "rgba(0,0,0,0)");
    vg.addColorStop(1, "rgba(0,0,0,0.45)");
    ctx.fillStyle = vg;
    ctx.fillRect(0, 0, W, H);
  }
  return c;
}

export default function GameCanvas({
  frame,
  onPickHero,
  showGhosts = true,
  view = DEFAULT_VIEW,
}: {
  frame: Frame | null;
  onPickHero?: (id: number) => void;
  showGhosts?: boolean;
  view?: ViewSettings;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [showLegend, setShowLegend] = useState(false);
  const curr = useRef<Frame | null>(null);
  const prev = useRef<Frame | null>(null);
  const arrivedAt = useRef<number>(0);
  // Measured inter-frame arrival cadence (EMA, ms). The nominal 1000/speed is
  // only a lower bound — the server sleeps AFTER step+snapshot+broadcast, so
  // real gaps are always nominal + work + network. Interpolating against the
  // measured gap keeps motion continuous instead of move-then-freeze.
  const gapEma = useRef<number>(0);
  const lastSpeed = useRef<number>(0);
  const foodSprite = useRef<HTMLCanvasElement | null>(null);
  const backdrop = useRef<HTMLCanvasElement | null>(null);
  const bdKey = useRef<string>("");

  // Mirror the ghost toggle into a ref the rAF loop can read each frame.
  const showGhostsRef = useRef(showGhosts);
  showGhostsRef.current = showGhosts;

  // Mirror view settings into a ref the rAF loop reads each frame.
  const viewRef = useRef(view);
  viewRef.current = view;

  // Live reduced-motion flag (updated on OS-setting change, no reload needed).
  const reducedRef = useRef(prefersReducedMotion());
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => (reducedRef.current = mq.matches);
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);

  // effect layers, all in native arena pixel coords
  const particles = useRef<Particle[]>([]);
  const pops = useRef<Map<number, number>>(new Map()); // foodKey -> born time
  const shake = useRef<{ mag: number; until: number }>({ mag: 0, until: 0 });
  const flash = useRef<{ r: number; g: number; b: number; born: number; dur: number } | null>(null);
  const lastTs = useRef<number>(0);

  // React to each new server frame: detect eat/death/kill/new-food events by
  // diffing against the frame it replaces, then advance the interpolation window.
  useEffect(() => {
    if (!frame) return;
    const old = curr.current;
    // Episode reset (frame counter regressed): drop the stale interpolation
    // window and any in-flight effects so snakes paint cleanly at spawn instead
    // of tweening from the previous episode. detectEvents suppresses bursts on the
    // same condition; here we also clear it so surviving-index snakes don't slide.
    if (old && frame.frame < old.frame) {
      prev.current = null;
      curr.current = frame;
      arrivedAt.current = performance.now();
      particles.current.length = 0;
      pops.current.clear();
      return;
    }
    detectEvents(old, frame, particles.current, pops.current, shake.current, flash);
    prev.current = old;
    curr.current = frame;
    const now = performance.now();
    // Update the arrival-cadence EMA from real frames only: skip paused
    // heartbeats, non-advancing frames, and stall outliers (tab hidden, network
    // hiccup). A speed change resets the EMA so it re-converges immediately.
    const speed = Math.max(1, Math.min(120, frame.session?.speed ?? 12));
    if (speed !== lastSpeed.current) {
      lastSpeed.current = speed;
      gapEma.current = 0;
    }
    const advanced = !old || frame.frame > old.frame;
    if (old && advanced && frame.session?.playing !== false && arrivedAt.current > 0) {
      const gap = now - arrivedAt.current;
      const nominal = 1000 / speed;
      if (gap > 0 && gap < nominal * 3 + 250) {
        gapEma.current = gapEma.current > 0 ? gapEma.current * 0.8 + gap * 0.2 : gap;
      }
    }
    arrivedAt.current = now;
  }, [frame]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      const dt = lastTs.current ? Math.min(0.05, (now - lastTs.current) / 1000) : 0.016;
      lastTs.current = now;
      const c = curr.current;
      if (!c) return;
      const reduced = reducedRef.current;
      const v = viewRef.current;

      const { arena } = c;
      const seg = arena.segment;
      const W = arena.width;
      const H = arena.height;
      // HiDPI: back the canvas at devicePixelRatio (clamped to bound memory)
      // while keeping the CSS size at the arena dimensions, so the board stays
      // sharp on retina screens and in fullscreen. Reading dpr each tick makes
      // monitor moves / zoom changes take effect without a reload.
      const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
      const bw = Math.round(W * dpr);
      const bh = Math.round(H * dpr);
      if (canvas.width !== bw) canvas.width = bw;
      if (canvas.height !== bh) canvas.height = bh;
      // Pin the CSS width to the arena size (what the intrinsic size was before
      // dpr scaling); height stays auto so max-width/max-height shrinking keeps
      // the aspect ratio exactly as it did pre-dpr.
      const cssW = `${W}px`;
      if (canvas.style.width !== cssW) canvas.style.width = cssW;

      if (foodSprite.current === null || (foodSprite.current as any)._seg !== seg) {
        foodSprite.current = makeFoodSprite(seg);
        (foodSprite.current as any)._seg = seg;
      }
      const key = `${W}x${H}x${seg}x${dpr}x${v.grid ? 1 : 0}x${v.vignette ? 1 : 0}`;
      if (bdKey.current !== key) {
        backdrop.current = makeBackdrop(W, H, seg, dpr, v.grid, v.vignette);
        bdKey.current = key;
      }

      const speed = Math.max(1, Math.min(120, c.session?.speed ?? 12));
      // Tween over the measured arrival cadence (EMA), not the nominal period —
      // the server always runs slower than nominal (it sleeps after doing the
      // work), so the nominal window makes every glide finish early and dwell.
      const nominal = 1000 / speed;
      const measured = gapEma.current;
      const interval =
        measured > 0 ? Math.max(nominal * 0.5, Math.min(nominal * 4, measured)) : nominal;
      const paused = c.session?.playing === false;
      const alpha = reduced || paused ? 1 : Math.max(0, Math.min(1, (now - arrivedAt.current) / interval));
      const t = now / 1000;

      // base fill so screen-shake never reveals an empty gap at the edges
      // (all drawing below happens in arena coordinates under the dpr transform)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#080b11";
      ctx.fillRect(0, 0, W, H);

      // screen-shake (Play deaths); decays to zero
      let sx = 0;
      let sy = 0;
      if (!reduced && shake.current.until > now) {
        const k = (shake.current.until - now) / 260;
        const m = shake.current.mag * k;
        sx = (Math.random() * 2 - 1) * m;
        sy = (Math.random() * 2 - 1) * m;
      }
      ctx.save();
      ctx.translate(sx, sy);

      if (backdrop.current) ctx.drawImage(backdrop.current, 0, 0, W, H);

      // --- food (with spawn-pop) -----------------------------------------
      const sprite = foodSprite.current;
      if (sprite) {
        // drop finished pops
        for (const [k, born] of pops.current) if (now - born > 260) pops.current.delete(k);
        for (const [fx, fy] of c.food) {
          const cx = fx + seg / 2;
          const cy = fy + seg / 2;
          const pulse = reduced ? 1 : 1 + 0.14 * Math.sin(t * 3 + (fx * 0.7 + fy * 0.9));
          const born = pops.current.get(foodKey(fx, fy));
          const pop = born != null ? 1 + 0.7 * (1 - Math.min(1, (now - born) / 260)) : 1;
          const s = seg * 2.0 * pulse * pop;
          ctx.drawImage(sprite, cx - s, cy - s, s * 2, s * 2);
        }
      }

      // --- snakes --------------------------------------------------------
      const prevById = new Map<number, SnakeDTO>();
      if (prev.current) for (const s of prev.current.snakes) prevById.set(s.id, s);
      const humanMode = c.session?.mode === "play";

      // interpolated segment centres per alive snake, shared by the ghost pass,
      // the ribbons, and the label/crown pass (computed once).
      const centersById = new Map<number, Pt[]>();
      for (const s of c.snakes) {
        if (s.alive) centersById.set(s.id, computeCenters(s, prevById.get(s.id), alpha, seg));
      }

      // intent ghosts: what the hero is considering (turn L/S/R), colored by the
      // danger of each move, sized by its Q-value, chosen one brightest. Drawn
      // under the snakes so the head sits on top of its own projection.
      if (showGhostsRef.current && !humanMode && c.inspector) {
        const hero = c.snakes.find((s) => s.is_hero && s.alive);
        const hc = hero ? centersById.get(hero.id) : undefined;
        if (hc && hc.length) drawIntentGhosts(ctx, hc, c.inspector, seg);
      }

      for (const snake of c.snakes) {
        if (!snake.alive) continue;
        const ce = centersById.get(snake.id);
        if (ce && ce.length) drawSnake(ctx, snake, ce, seg, humanMode, t, reduced, v.trails);
      }

      // floating name labels + a crown on the current leader
      if (v.labels || v.crown) drawLabelsAndCrown(ctx, c.snakes, centersById, seg, v);

      // --- particles (additive glow) -------------------------------------
      const ps = particles.current;
      if (ps.length) {
        ctx.globalCompositeOperation = "lighter";
        for (let i = ps.length - 1; i >= 0; i--) {
          const p = ps[i];
          const age = (now - p.born) / 1000;
          if (age >= p.life) {
            ps.splice(i, 1);
            continue;
          }
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          p.vx *= 1 - 2.2 * dt;
          p.vy *= 1 - 2.2 * dt;
          const a = 1 - age / p.life;
          ctx.globalAlpha = a;
          ctx.fillStyle = `rgb(${p.color[0]},${p.color[1]},${p.color[2]})`;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * (0.5 + a * 0.5), 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = "source-over";
      }

      ctx.restore();

      // --- full-screen flash (Play kill/death) ---------------------------
      const fl = flash.current;
      if (fl) {
        const age = (now - fl.born) / fl.dur;
        if (age >= 1) {
          flash.current = null;
        } else {
          ctx.fillStyle = `rgba(${fl.r},${fl.g},${fl.b},${0.35 * (1 - age)})`;
          ctx.fillRect(0, 0, W, H);
        }
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Click a snake to make it the hero/inspected one. Maps the click through the
  // CSS scale into arena pixels (the backing store is dpr-scaled, so use the
  // arena dims, not canvas.width), then hit-tests against the live segment
  // centres. Disabled in Play mode where the hero is always "you".
  const onClick = (e: ReactMouseEvent<HTMLCanvasElement>) => {
    const c = curr.current;
    const canvas = ref.current;
    if (!c || !canvas || !onPickHero || c.session?.mode === "play") return;
    const rect = canvas.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * c.arena.width;
    const ny = ((e.clientY - rect.top) / rect.height) * c.arena.height;
    const seg = c.arena.segment;
    let best: number | null = null;
    let bestD = (seg * 2.2) ** 2;
    for (const s of c.snakes) {
      if (!s.alive) continue;
      for (const [x, y] of s.segments) {
        const dx = x + seg / 2 - nx;
        const dy = y + seg / 2 - ny;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          best = s.id;
        }
      }
    }
    if (best != null) onPickHero(best);
  };

  // Save the current arena frame as a PNG (share a cool moment / a run).
  const savePng = () => {
    const c = ref.current;
    if (!c) return;
    const a = document.createElement("a");
    a.download = `snake-dqn-${curr.current?.frame ?? 0}.png`;
    a.href = c.toDataURL("image/png");
    a.click();
  };

  // Cinematic view: pop the arena to fullscreen (Esc or the button exits).
  const toggleFullscreen = () => {
    const stage = ref.current?.parentElement;
    if (!stage) return;
    if (document.fullscreenElement) document.exitFullscreen?.();
    else stage.requestFullscreen?.();
  };

  const pickable = !!onPickHero && frame?.session?.mode !== "play";

  // Text alternative for the canvas: a short frame summary used as both the
  // accessible name and a visually-hidden description line. Deliberately NOT an
  // aria-live region — frames arrive ~12x/sec and announcing each would be noise;
  // screen readers read it on demand instead.
  let ariaSummary = "Live snake arena — waiting for the server";
  if (frame) {
    const aliveSnakes = frame.snakes.filter((s) => s.alive);
    let leader: SnakeDTO | null = null;
    for (const s of aliveSnakes) if (!leader || s.length > leader.length) leader = s;
    const mode = frame.session?.mode ?? "watch";
    const paused = frame.session?.playing === false;
    ariaSummary =
      `Live snake arena, ${mode} mode${paused ? " (paused)" : ""}: ` +
      `${aliveSnakes.length} of ${frame.snakes.length} snakes alive` +
      (leader ? `, leader ${leader.name} at ${leader.length}` : "");
  }

  return (
    <div className="stage">
      <canvas
        ref={ref}
        onClick={onClick}
        className={pickable ? "pickable" : undefined}
        title={pickable ? "Click a snake to inspect it" : undefined}
        role="img"
        aria-label={ariaSummary}
      />
      <p
        className="visually-hidden"
        style={{
          position: "absolute",
          width: 1,
          height: 1,
          padding: 0,
          margin: -1,
          overflow: "hidden",
          clip: "rect(0 0 0 0)",
          whiteSpace: "nowrap",
          border: 0,
        }}
      >
        {ariaSummary}
        {pickable ? " Use the roster in Controls, or the [ and ] keys, to inspect a snake." : ""}
      </p>
      <div className="stage-controls">
        <button
          className={"stage-btn" + (showLegend ? " on" : "")}
          onClick={() => setShowLegend((v) => !v)}
          title="What am I looking at? (arena legend)"
          aria-label="Arena legend"
          aria-expanded={showLegend}
        >
          ⓘ
        </button>
        <button className="stage-btn" onClick={savePng} title="Save arena as PNG" aria-label="Save arena as PNG">
          ⤓
        </button>
        <button
          className="stage-btn"
          onClick={toggleFullscreen}
          title="Fullscreen arena"
          aria-label="Fullscreen arena"
        >
          ⛶
        </button>
      </div>
      {showLegend && <ArenaLegend onClose={() => setShowLegend(false)} />}
      {frame && (
        <EpisodeSummary
          stats={frame.stats ?? null}
          frameNum={frame.frame}
          mode={frame.session?.mode ?? "watch"}
        />
      )}
      {frame?.session?.mode !== "play" && (
        <DecisionNarrator inspector={frame?.inspector ?? null} obsSpec={frame?.obs_spec} />
      )}
      {frame?.session?.playing === false && frame?.session?.mode !== "play" && (
        <div className="pause-scrim" role="status">
          <div className="pause-card">
            <span className="pause-glyph">❚❚</span>
            <span>Paused</span>
            <span className="pause-hint">
              press <span className="kbd">Space</span> to resume
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// Diff two consecutive frames and spawn effects. Zero backend change: eat is a
// per-snake length increase, death is alive true->false, new food is a fresh
// pellet key, Play kills/deaths come from play state.
function detectEvents(
  old: Frame | null,
  next: Frame,
  particles: Particle[],
  pops: Map<number, number>,
  shake: { mag: number; until: number },
  flashRef: { current: { r: number; g: number; b: number; born: number; dur: number } | null }
) {
  if (!old || prefersReducedMotion()) return;
  // reset / new episode: frame counter went backwards — suppress the burst storm
  if (next.frame < old.frame) return;
  const now = performance.now();
  const seg = next.arena.segment;

  const oldById = new Map<number, SnakeDTO>();
  for (const s of old.snakes) oldById.set(s.id, s);

  for (const s of next.snakes) {
    const o = oldById.get(s.id);
    if (!o) continue;
    const head: Pt = [s.head[0] + seg / 2, s.head[1] + seg / 2];
    const teleport = Math.hypot(s.head[0] - o.head[0], s.head[1] - o.head[1]) > seg * 3;
    if (s.alive && o.alive && s.length > o.length && !teleport) {
      burst(particles, head, [255, 210, 110], 8, 70, 150, 0.45, seg, now);
    }
    if (o.alive && !s.alive) {
      const oh: Pt = [o.head[0] + seg / 2, o.head[1] + seg / 2];
      const col = bodyColors(o.color).lifted;
      burst(particles, oh, [col[0] | 0, col[1] | 0, col[2] | 0], 16, 90, 240, 0.75, seg, now);
    }
  }

  // new food pellets -> spawn-pop
  const oldFood = new Set<number>();
  for (const [x, y] of old.food) oldFood.add(foodKey(x, y));
  let added = 0;
  for (const [x, y] of next.food) {
    const k = foodKey(x, y);
    if (!oldFood.has(k)) {
      pops.set(k, now);
      if (++added > 40) break; // guard: a reset can add hundreds at once
    }
  }

  // Play-mode screen feedback
  if (next.play && old.play) {
    if (next.play.kills > old.play.kills) {
      flashRef.current = { r: 240, g: 255, b: 245, born: now, dur: 180 };
    }
    if (old.play.human_alive && !next.play.human_alive) {
      flashRef.current = { r: 248, g: 90, b: 90, born: now, dur: 420 };
      shake.mag = 9;
      shake.until = now + 260;
    }
  }
}

function burst(
  out: Particle[],
  [x, y]: Pt,
  color: [number, number, number],
  n: number,
  vmin: number,
  vmax: number,
  life: number,
  seg: number,
  now: number
) {
  for (let i = 0; i < n; i++) {
    const ang = Math.random() * Math.PI * 2;
    const v = vmin + Math.random() * (vmax - vmin);
    out.push({
      x,
      y,
      vx: Math.cos(ang) * v,
      vy: Math.sin(ang) * v,
      born: now,
      life: life * (0.7 + Math.random() * 0.6),
      size: seg * (0.18 + Math.random() * 0.22),
      color,
    });
  }
  if (out.length > 600) out.splice(0, out.length - 600);
}

// Interpolated segment centres (px, at cell centre) between prev and curr. A big
// jump (teleport / respawn) disables interpolation so nothing streaks across.
function computeCenters(
  snake: SnakeDTO,
  p: SnakeDTO | undefined,
  alpha: number,
  seg: number
): Pt[] {
  const segs = snake.segments;
  if (segs.length === 0) return [];
  let useInterp = !!p && p.alive && p.segments.length > 0;
  if (useInterp && p) {
    const dx = segs[0][0] - p.segments[0][0];
    const dy = segs[0][1] - p.segments[0][1];
    if (Math.hypot(dx, dy) > seg * 3) useInterp = false;
  }
  const centers: Pt[] = new Array(segs.length);
  for (let i = 0; i < segs.length; i++) {
    const [cx, cy] = segs[i];
    if (useInterp && p && p.segments[i]) {
      const [px, py] = p.segments[i];
      centers[i] = [lerp(px, cx, alpha) + seg / 2, lerp(py, cy, alpha) + seg / 2];
    } else {
      centers[i] = [cx + seg / 2, cy + seg / 2];
    }
  }
  return centers;
}

// green (safe) -> amber -> red (dangerous), for the per-action danger of a move.
function dangerHeat(d: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, d));
  const a: [number, number, number] = [52, 211, 153];
  const b: [number, number, number] = [251, 191, 36];
  const c: [number, number, number] = [248, 113, 113];
  const [lo, hi, f] = x < 0.5 ? [a, b, x / 0.5] : [b, c, (x - 0.5) / 0.5];
  return [lo[0] + (hi[0] - lo[0]) * f, lo[1] + (hi[1] - lo[1]) * f, lo[2] + (hi[2] - lo[2]) * f];
}

// Project the hero's three candidate moves (turn-left / straight / turn-right) as
// translucent cones from its head: length scales with the move's Q-value, colour
// with its danger, the chosen move brightest. A read of "what it's considering".
function drawIntentGhosts(
  ctx: CanvasRenderingContext2D,
  centers: Pt[],
  inspector: NonNullable<Frame["inspector"]>,
  seg: number
): void {
  const q = inspector.q_values;
  if (!q || q.length < 3) return;
  const head = centers[0];
  let fx = 1;
  let fy = 0;
  if (centers.length > 1) {
    fx = head[0] - centers[1][0];
    fy = head[1] - centers[1][1];
    const m = Math.hypot(fx, fy) || 1;
    fx /= m;
    fy /= m;
  }
  const st = inspector.state;
  const danger = st.length > 56 ? [st[54], st[55], st[56]] : [0, 0, 0];
  const q3 = [q[0], q[1], q[2]];
  const mn = Math.min(...q3);
  const rng = Math.max(...q3) - mn || 1;
  const chosenDir = inspector.chosen % 3;
  const rot = (vx: number, vy: number, a: number): Pt => [
    vx * Math.cos(a) - vy * Math.sin(a),
    vx * Math.sin(a) + vy * Math.cos(a),
  ];
  const THETA = 0.62;
  const dirs: Pt[] = [rot(fx, fy, -THETA), [fx, fy], rot(fx, fy, THETA)];

  ctx.save();
  ctx.lineJoin = "round";
  for (let i = 0; i < 3; i++) {
    const [dx, dy] = dirs[i];
    const len = seg * (3.4 + 5.6 * ((q3[i] - mn) / rng));
    const halfW = seg * 0.62;
    const bx = head[0] + dx * len;
    const by = head[1] + dy * len;
    const px = -dy;
    const py = dx;
    const [r, g, b] = dangerHeat(danger[i]);
    const chosen = i === chosenDir;
    const rgb = `${r | 0},${g | 0},${b | 0}`;
    ctx.beginPath();
    ctx.moveTo(head[0], head[1]);
    ctx.lineTo(bx + px * halfW, by + py * halfW);
    ctx.lineTo(bx - px * halfW, by - py * halfW);
    ctx.closePath();
    ctx.fillStyle = `rgba(${rgb},${chosen ? 0.4 : 0.15})`;
    ctx.fill();
    ctx.strokeStyle = `rgba(${rgb},${chosen ? 0.85 : 0.4})`;
    ctx.lineWidth = chosen ? 1.6 : 1;
    ctx.stroke();
    // soft rounded tip so each move reads as an arrow, not a spike
    ctx.beginPath();
    ctx.arc(bx, by, halfW * 0.5, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${rgb},${chosen ? 0.55 : 0.22})`;
    ctx.fill();
  }
  ctx.restore();
}

// Floating name labels above each snake's head, and a gold crown on the current
// leader (longest alive snake). Cheap: one text draw + one small path per snake.
function drawLabelsAndCrown(
  ctx: CanvasRenderingContext2D,
  snakes: SnakeDTO[],
  centersById: Map<number, Pt[]>,
  seg: number,
  view: ViewSettings
) {
  let leaderId = -1;
  let leaderLen = -1;
  for (const s of snakes) {
    if (s.alive && s.length > leaderLen) {
      leaderLen = s.length;
      leaderId = s.id;
    }
  }
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  const fs = Math.max(9, seg * 0.95);
  ctx.font = `600 ${fs}px system-ui, -apple-system, sans-serif`;
  for (const s of snakes) {
    if (!s.alive) continue;
    const ce = centersById.get(s.id);
    if (!ce || !ce.length) continue;
    const [hx, hy] = ce[0];
    let y = hy - seg * 1.2;
    if (view.crown && s.id === leaderId) {
      drawCrown(ctx, hx, y, seg);
      y -= seg * 1.1;
    }
    if (view.labels) {
      ctx.lineWidth = Math.max(2.2, seg * 0.18);
      ctx.strokeStyle = "rgba(6,9,14,0.9)";
      ctx.strokeText(s.name, hx, y);
      ctx.fillStyle = "#e6edf6";
      ctx.fillText(s.name, hx, y);
    }
  }
  ctx.restore();
}

// A small three-spike crown centred horizontally on x, sitting with its base at y.
function drawCrown(ctx: CanvasRenderingContext2D, x: number, y: number, seg: number) {
  const w = seg * 1.5;
  const h = seg * 0.95;
  const left = x - w / 2;
  const base = y;
  const top = y - h;
  ctx.beginPath();
  ctx.moveTo(left, base);
  ctx.lineTo(left, top + h * 0.4);
  ctx.lineTo(left + w * 0.16, top);
  ctx.lineTo(left + w * 0.34, top + h * 0.5);
  ctx.lineTo(x, top - h * 0.15);
  ctx.lineTo(left + w * 0.66, top + h * 0.5);
  ctx.lineTo(left + w * 0.84, top);
  ctx.lineTo(left + w, top + h * 0.4);
  ctx.lineTo(left + w, base);
  ctx.closePath();
  ctx.fillStyle = "#ffd25e";
  ctx.strokeStyle = "rgba(90,60,0,0.6)";
  ctx.lineWidth = 1;
  ctx.fill();
  ctx.stroke();
}

// Paint a snake from precomputed segment centres as a smooth rounded ribbon with
// a head->tail gradient, a glossy head and eyes.
function drawSnake(
  ctx: CanvasRenderingContext2D,
  snake: SnakeDTO,
  centers: Pt[],
  seg: number,
  humanMode: boolean,
  t: number,
  reduced: boolean,
  trails: boolean
) {
  if (centers.length === 0) return;

  const { bright, dim } = bodyColors(snake.color);
  const head = centers[0];
  const tail = centers[centers.length - 1];

  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // boost trail: a translucent, fatter ribbon underneath for a sense of speed
  if (snake.boosting && !reduced && trails) {
    ctx.globalAlpha = 0.22;
    ctx.strokeStyle = bright;
    ctx.lineWidth = seg * 1.55;
    ribbon(ctx, centers);
    ctx.globalAlpha = 1;
  }

  // body ribbon with a head->tail gradient
  const grad = ctx.createLinearGradient(head[0], head[1], tail[0], tail[1]);
  grad.addColorStop(0, bright);
  grad.addColorStop(1, dim);
  if (snake.boosting) {
    ctx.shadowColor = bright;
    ctx.shadowBlur = 16;
  }
  ctx.strokeStyle = grad;
  ctx.lineWidth = seg * 0.92;
  ribbon(ctx, centers);
  ctx.shadowBlur = 0;

  // hero ring: the snake you're watching (or "you" in play mode)
  if (snake.is_hero) {
    if (humanMode) {
      ctx.strokeStyle = "#7dd3fc";
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 14 + (reduced ? 0 : 4 * Math.sin(t * 4));
      ctx.lineWidth = 2.5;
    } else {
      ctx.strokeStyle = "rgba(230,237,246,0.9)";
      ctx.shadowBlur = 0;
      ctx.lineWidth = 2;
    }
    ctx.beginPath();
    ctx.arc(head[0], head[1], seg * 0.72, 0, Math.PI * 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // glossy head cap
  ctx.fillStyle = bright;
  ctx.beginPath();
  ctx.arc(head[0], head[1], seg * 0.55, 0, Math.PI * 2);
  ctx.fill();

  drawEyes(ctx, centers, seg);
  ctx.restore();
}

function ribbon(ctx: CanvasRenderingContext2D, centers: Pt[]) {
  ctx.beginPath();
  ctx.moveTo(centers[0][0], centers[0][1]);
  if (centers.length === 1) {
    ctx.lineTo(centers[0][0] + 0.01, centers[0][1]);
  } else {
    for (let i = 1; i < centers.length; i++) ctx.lineTo(centers[i][0], centers[i][1]);
  }
  ctx.stroke();
}

function drawEyes(ctx: CanvasRenderingContext2D, centers: Pt[], seg: number) {
  const head = centers[0];
  let dx = 1;
  let dy = 0;
  if (centers.length > 1) {
    dx = head[0] - centers[1][0];
    dy = head[1] - centers[1][1];
    const m = Math.hypot(dx, dy) || 1;
    dx /= m;
    dy /= m;
  }
  const perp: Pt = [-dy, dx];
  const fwd = seg * 0.16;
  const side = seg * 0.24;
  const eyeR = Math.max(1.1, seg * 0.17);
  const pupilR = Math.max(0.6, seg * 0.09);
  for (const sgn of [1, -1]) {
    const ex = head[0] + dx * fwd + perp[0] * side * sgn;
    const ey = head[1] + dy * fwd + perp[1] * side * sgn;
    ctx.fillStyle = "#f8fbff";
    ctx.beginPath();
    ctx.arc(ex, ey, eyeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#0b1220";
    ctx.beginPath();
    ctx.arc(ex + dx * pupilR, ey + dy * pupilR, pupilR, 0, Math.PI * 2);
    ctx.fill();
  }
}
