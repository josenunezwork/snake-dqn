import { useCallback, useEffect, useRef, useState } from "react";
import { wsUrl } from "../api";
import type { ControlAction, Frame, MetricSample } from "../types";

// A frozen board looks identical to a live one, so we surface link health
// explicitly: connected + fresh frames = live; connected but the (playing) engine
// stopped streaming = stale; dropped after a prior connection = reconnecting.
export type LinkStatus = "live" | "stale" | "reconnecting" | "offline";

interface SocketState {
  frame: Frame | null;
  connected: boolean;
  link: LinkStatus;
  // Rolling window of recent per-frame metrics for the live sparklines. Reset
  // whenever the loaded checkpoint changes so a stale-model trend never bleeds
  // into a freshly loaded one.
  history: MetricSample[];
  send: (action: ControlAction, value?: unknown) => void;
}

const STALE_MS = 2500;

const HISTORY_CAP = 240; // ~20s at 12fps; bounded so long sessions don't grow unbounded

// Opens the frame-stream WebSocket, keeps the latest frame in state, accumulates
// a bounded metric history, and exposes a control sender. Auto-reconnects on drop.
export function useGameSocket(): SocketState {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [connected, setConnected] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number | null>(null);
  const historyRef = useRef<MetricSample[]>([]);
  const lastCheckpointRef = useRef<string | null>(null);
  const lastFrameAtRef = useRef<number>(0);
  const everConnectedRef = useRef<boolean>(false);

  useEffect(() => {
    let closed = false;

    const connect = () => {
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        everConnectedRef.current = true;
        setConnected(true);
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retryRef.current = window.setTimeout(connect, 1000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg && msg.type === "frame") {
            lastFrameAtRef.current = Date.now();
            accumulate(historyRef, lastCheckpointRef, msg as Frame);
            setFrame(msg as Frame);
          }
        } catch {
          /* ignore malformed */
        }
      };
    };

    connect();
    // Heartbeat so link staleness is re-evaluated even when no frames arrive.
    const heartbeat = window.setInterval(() => setNow(Date.now()), 1000);
    return () => {
      closed = true;
      window.clearInterval(heartbeat);
      if (retryRef.current) window.clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((action: ControlAction, value?: unknown) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "control", action, value }));
    }
  }, []);

  // A paused game legitimately stops streaming — only flag "stale" when the last
  // frame we saw said the engine was playing.
  const playing = frame?.session?.playing !== false;
  let link: LinkStatus;
  if (!connected) link = everConnectedRef.current ? "reconnecting" : "offline";
  else if (frame && playing && lastFrameAtRef.current && now - lastFrameAtRef.current > STALE_MS)
    link = "stale";
  else link = "live";

  return { frame, connected, link, history: historyRef.current, send };
}

function accumulate(
  historyRef: React.MutableRefObject<MetricSample[]>,
  lastCheckpointRef: React.MutableRefObject<string | null>,
  frame: Frame
) {
  const s = frame.stats;
  if (!s) return;
  const ckpt = frame.session?.checkpoint ?? null;
  if (ckpt !== lastCheckpointRef.current) {
    historyRef.current = [];
    lastCheckpointRef.current = ckpt;
  }
  const hist = historyRef.current;
  hist.push({
    frame: s.frame,
    loss: s.loss ?? null,
    epsilon: s.epsilon,
    foodEaten: s.food_eaten,
    alive: s.alive,
    bestLength: s.best_length,
    kills: s.kills,
  });
  if (hist.length > HISTORY_CAP) hist.splice(0, hist.length - HISTORY_CAP);
}
