import { useEffect, useRef } from "react";

const COLORS = ["#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#f87171"];

// Dependency-free celebratory confetti: a full-screen, click-through canvas that
// runs a single ~2.6s burst then calls onDone so the parent can unmount it.
// Honors prefers-reduced-motion by skipping straight to onDone.
export default function Confetti({ onDone }: { onDone?: () => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  // Keep the latest onDone in a ref so the animation effect can run exactly once
  // on mount ([] deps). The parent passes an inline onDone that changes identity
  // every frame; depending on it would restart the burst ~12x/sec (never
  // finishing, never clearing).
  const onDoneRef = useRef(onDone);
  useEffect(() => {
    onDoneRef.current = onDone;
  });
  useEffect(() => {
    const reduce =
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const canvas = ref.current;
    if (reduce || !canvas) {
      onDoneRef.current?.();
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      onDoneRef.current?.();
      return;
    }
    const W = (canvas.width = window.innerWidth);
    const H = (canvas.height = window.innerHeight);
    const parts = Array.from({ length: 150 }, () => ({
      x: W / 2 + (Math.random() - 0.5) * W * 0.5,
      y: H * 0.28 + (Math.random() - 0.5) * 40,
      vx: (Math.random() - 0.5) * 9,
      vy: -7 - Math.random() * 7,
      g: 0.2 + Math.random() * 0.12,
      size: 4 + Math.random() * 5,
      color: COLORS[(Math.random() * COLORS.length) | 0],
      rot: Math.random() * Math.PI * 2,
      vr: (Math.random() - 0.5) * 0.45,
    }));

    let raf = 0;
    const start = performance.now();
    const draw = (now: number) => {
      const t = now - start;
      if (t > 2600) {
        cancelAnimationFrame(raf);
        onDoneRef.current?.();
        return;
      }
      ctx.clearRect(0, 0, W, H);
      const fade = t > 1800 ? Math.max(0, 1 - (t - 1800) / 800) : 1;
      for (const p of parts) {
        p.vy += p.g;
        p.x += p.vx;
        p.y += p.vy;
        p.rot += p.vr;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.globalAlpha = fade;
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
        ctx.restore();
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={ref} className="confetti-overlay" aria-hidden="true" />;
}
