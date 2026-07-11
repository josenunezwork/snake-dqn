import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import GameCanvas from "./components/GameCanvas";
import Inspector from "./components/Inspector";
import EgoRasterViewer from "./components/EgoRasterViewer";
import NetworkVisualizer from "./components/NetworkVisualizer";
import Dashboard from "./components/Dashboard";
import Controls from "./components/Controls";
import Play from "./components/Play";
import TelemetryBand from "./components/TelemetryBand";
import HelpOverlay from "./components/HelpOverlay";
import CommandPalette, { type Command } from "./components/CommandPalette";
import WelcomeTour from "./components/WelcomeTour";
import ViewMenu from "./components/ViewMenu";
import Toast from "./components/Toast";
import { useGameSocket } from "./hooks/useGameSocket";
import { useViewSettings } from "./hooks/useViewSettings";
import { fetchCheckpoints } from "./api";
import type { CheckpointInfo, SnakeDTO } from "./types";
import { SPEED_PRESETS, activePresetIndex } from "./lib/presets";
import { runSummary, copyText } from "./lib/runSummary";
import { resolveMoveKey } from "./keys";

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
const CMDK_LABEL = IS_MAC ? "⌘K" : "Ctrl K";

type Tab = "controls" | "play" | "inspector" | "raster" | "network" | "dashboard";

const TABS: { id: Tab; label: string }[] = [
  { id: "controls", label: "Controls" },
  { id: "play", label: "Play" },
  { id: "inspector", label: "Inspector" },
  { id: "raster", label: "Raster" },
  { id: "network", label: "Network" },
  { id: "dashboard", label: "Dashboard" },
];

// Restore the last tab, but never auto-enter Play (it rebuilds the game around a
// human snake) — reopen on Controls instead.
function initialTab(): Tab {
  const saved = localStorage.getItem("snake_tab");
  if (saved && saved !== "play" && TABS.some((t) => t.id === saved)) return saved as Tab;
  return "controls";
}

export default function App() {
  const { frame, link, history, send } = useGameSocket();
  const [tab, setTab] = useState<Tab>(initialTab);
  const [showHelp, setShowHelp] = useState(false);
  const [showPalette, setShowPalette] = useState(false);
  const [showTour, setShowTour] = useState(() => !localStorage.getItem("snake_onboarded"));
  const [showGhosts, setShowGhosts] = useState(() => localStorage.getItem("snake_ghosts") !== "0");
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [toast, setToast] = useState<{ msg: string; tone: "error" | "ok" } | null>(null);
  const [view, updateView] = useViewSettings();
  const lastErrorRef = useRef<string | null>(null);
  const pendingModeRef = useRef<string | null>(null);
  // Latest snakes + hero for keyboard hero-cycling, without re-binding the
  // listener every frame.
  const heroCycleRef = useRef<{ snakes: SnakeDTO[]; heroId: number }>({ snakes: [], heroId: 0 });
  heroCycleRef.current = { snakes: frame?.snakes ?? [], heroId: frame?.session?.hero_id ?? 0 };
  const stats = frame?.stats;
  const session = frame?.session;
  const playing = session?.mode === "play";
  // Fresh in the key handler's stable closure: whether the human run has ended.
  const runOverRef = useRef(false);
  runOverRef.current = !!frame?.play?.run_over;

  // Entering the Play tab switches the engine into play mode; leaving reverts to
  // watch. The server rebuilds the game around a human snake on this signal.
  useEffect(() => {
    if (tab === "play") {
      send("set_mode", "play");
      return () => send("set_mode", "watch");
    }
  }, [tab, send]);

  // Remember the last non-Play tab across reloads.
  useEffect(() => {
    if (tab !== "play") localStorage.setItem("snake_tab", tab);
  }, [tab]);

  useEffect(() => {
    localStorage.setItem("snake_ghosts", showGhosts ? "1" : "0");
  }, [showGhosts]);

  // Checkpoint inventory for the command palette (fetched once; refreshed when a
  // load succeeds and the active checkpoint changes).
  useEffect(() => {
    fetchCheckpoints()
      .then(setCheckpoints)
      .catch(() => setCheckpoints([]));
  }, [session?.checkpoint]);

  // Switching to watch/train from the Play tab must first *leave* the tab (whose
  // cleanup reverts the engine to watch); we then apply the wanted mode once the
  // tab has changed, so a "Train" command from Play doesn't get clobbered.
  useEffect(() => {
    if (tab !== "play" && pendingModeRef.current) {
      send("set_mode", pendingModeRef.current);
      pendingModeRef.current = null;
    }
  }, [tab, send]);

  const replayTour = useCallback(() => {
    localStorage.removeItem("snake_onboarded");
    setShowTour(true);
  }, []);

  // One shared, tracked auto-dismiss timer so the error toast and the "copied ✓"
  // toast never clobber each other's lifetime.
  const toastTimerRef = useRef<number | null>(null);
  const showToast = useCallback((msg: string, tone: "error" | "ok", ms: number) => {
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    setToast({ msg, tone });
    toastTimerRef.current = window.setTimeout(() => setToast(null), ms);
  }, []);
  const dismissToast = useCallback(() => {
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    setToast(null);
  }, []);
  useEffect(() => () => {
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
  }, []);

  const copySummary = useCallback(async () => {
    const ok = await copyText(runSummary(frame));
    showToast(ok ? "Run summary copied ✓" : "Copy failed", ok ? "ok" : "error", 2500);
  }, [frame, showToast]);

  const switchMode = useCallback(
    (m: "watch" | "train" | "play") => {
      if (m === "play") {
        setTab("play");
        return;
      }
      if (tab === "play") {
        pendingModeRef.current = m;
        setTab("controls");
      } else {
        send("set_mode", m);
      }
    },
    [tab, send]
  );

  // Command-palette actions, rebuilt only while open (keeps the hero/model lists
  // fresh without allocating a command list on every closed frame).
  const commands = useMemo<Command[]>(() => {
    if (!showPalette || !session || !frame) return [];
    const cmds: Command[] = [
      {
        id: "toggle-play",
        title: session.playing ? "Pause" : "Play",
        group: "Playback",
        hint: "Space",
        run: () => send(session.playing ? "pause" : "play"),
      },
      { id: "reset", title: "Reset game", group: "Playback", hint: "R", run: () => send("reset") },
      {
        id: "mode-watch",
        title: "Watch mode",
        group: "Mode",
        active: session.mode === "watch",
        run: () => switchMode("watch"),
      },
      {
        id: "mode-train",
        title: "Train mode — learn live",
        group: "Mode",
        keywords: "learning",
        active: session.mode === "train",
        run: () => switchMode("train"),
      },
      {
        id: "mode-play",
        title: "Race the AI — Play mode",
        group: "Mode",
        keywords: "human versus",
        active: session.mode === "play",
        run: () => switchMode("play"),
      },
      {
        id: "intent",
        title: showGhosts ? "Hide intent overlay" : "Show intent overlay",
        group: "View",
        hint: "O",
        active: showGhosts,
        run: () => setShowGhosts((g) => !g),
      },
      { id: "help", title: "Keyboard shortcuts", group: "View", hint: "?", run: () => setShowHelp(true) },
      { id: "tour", title: "Replay welcome tour", group: "View", keywords: "onboarding help intro", run: replayTour },
      { id: "copy", title: "Copy run summary", group: "View", keywords: "clipboard export share log stats", run: copySummary },
    ];
    const ai = activePresetIndex(session.speed);
    SPEED_PRESETS.forEach((p, i) =>
      cmds.push({
        id: `speed-${p.fps}`,
        title: `Speed ${p.label} (${p.fps} fps)`,
        group: "Speed",
        hint: String(i + 1),
        keywords: `${p.label.replace("×", "x")} fast slow`,
        active: ai === i,
        run: () => send("set_speed", p.fps),
      })
    );
    [0, 0.05, 0.1, 0.25, 0.5, 1].forEach((e) =>
      cmds.push({
        id: `eps-${e}`,
        title: `Epsilon ${e.toFixed(2)}`,
        group: "Exploration",
        keywords: "randomness explore",
        run: () => send("set_epsilon", e),
      })
    );
    [50, 100, 200, 300, 600].forEach((f) =>
      cmds.push({
        id: `food-${f}`,
        title: `Food target ${f}`,
        group: "Food",
        run: () => send("set_food", f),
      })
    );
    TABS.forEach((t) =>
      cmds.push({
        id: `tab-${t.id}`,
        title: `Go to ${t.label}`,
        group: "Panel",
        active: tab === t.id,
        run: () => setTab(t.id),
      })
    );
    if (session.mode !== "play") {
      for (const s of frame.snakes) {
        cmds.push({
          id: `hero-${s.id}`,
          title: `Inspect ${s.name}${s.alive ? "" : " (dead)"}`,
          group: "Inspect",
          keywords: `snake ${s.id} hero focus`,
          active: s.id === session.hero_id,
          run: () => {
            send("set_hero", s.id);
            setTab((cur) => (cur === "network" ? cur : "inspector"));
          },
        });
      }
    }
    for (const c of checkpoints) {
      cmds.push({
        id: `ckpt-${c.name}`,
        title: `Load ${c.name}`,
        group: "Model",
        keywords: "checkpoint weights",
        active: c.name === session.checkpoint,
        run: () => send("load_checkpoint", c.name),
      });
    }
    return cmds;
  }, [showPalette, session, frame, tab, checkpoints, showGhosts, send, switchMode, replayTour, copySummary]);

  // Surface session errors (bad checkpoint / mode swap) as a transient toast,
  // firing only on the transition to a new error (it persists across frames).
  const sessionError = session?.error ?? null;
  useEffect(() => {
    if (sessionError && sessionError !== lastErrorRef.current) {
      showToast(sessionError, "error", 5000);
      lastErrorRef.current = sessionError;
    }
    if (!sessionError) lastErrorRef.current = null;
  }, [sessionError, showToast]);

  // Keyboard: in play mode, arrows/WASD steer and Space boosts. Otherwise the
  // classic Space = play/pause, R = reset shortcuts apply. Ignored while typing.
  useEffect(() => {
    const isTyping = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
    };
    const onKeyDown = (e: KeyboardEvent) => {
      // Command palette toggles from anywhere, even inside inputs (⌘K / Ctrl-K) —
      // but not while the first-run tour is up (it owns the keyboard).
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (!showTour) setShowPalette((p) => !p);
        return;
      }
      // Never hijack browser/OS accelerators (⌘R, ⌘1-5, ⌥…). ⌘K is handled above;
      // Shift is allowed through since "?" needs it.
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (showPalette) return; // the palette owns the keyboard while open
      if (showTour) return; // the tour owns the keyboard while open
      if (isTyping(e)) return;
      if (e.key === "Escape") {
        if (showHelp) {
          setShowHelp(false);
          e.preventDefault();
        }
        return;
      }
      if (e.key === "?") {
        setShowHelp((h) => !h);
        e.preventDefault();
        return;
      }
      if (showHelp) return; // the help modal swallows other shortcuts while open
      // Space on a focused button in watch/train: let the button's own activation
      // handle it so we don't ALSO toggle play/pause (double-fire). In PLAY mode we
      // must NOT skip — Space is boost, and the Play tab button often still holds
      // focus, which would otherwise make boost dead until the user clicks away.
      const tag = (e.target as HTMLElement)?.tagName;
      if (!playing && e.code === "Space" && tag === "BUTTON") return;
      if (playing) {
        // After a run ends, R or Enter starts a fresh game instantly.
        if (runOverRef.current && (e.key === "r" || e.key === "R" || e.key === "Enter")) {
          e.preventDefault();
          send("new_game");
          return;
        }
        const dir = resolveMoveKey(e.key);
        if (dir) {
          e.preventDefault();
          send("human_input", dir);
        } else if (e.code === "Space") {
          e.preventDefault();
          // hold-to-boost: send once on press, not on every auto-repeat keydown
          if (!e.repeat) send("human_boost", true);
        }
        return;
      }
      if (e.repeat) return; // holding Space/R must not spam pause/reset
      if (e.code === "Space") {
        e.preventDefault();
        send(session?.playing ? "pause" : "play");
      } else if (e.key.toLowerCase() === "r") {
        send("reset");
      } else if (e.key.toLowerCase() === "o") {
        e.preventDefault();
        setShowGhosts((g) => !g);
      } else if (e.key >= "1" && e.key <= "5") {
        const p = SPEED_PRESETS[Number(e.key) - 1];
        if (p) {
          e.preventDefault();
          send("set_speed", p.fps);
        }
      } else if (e.key === "[" || e.key === "]") {
        // cycle the inspected snake, alive-first / longest-first (roster order)
        e.preventDefault();
        const { snakes, heroId } = heroCycleRef.current;
        const order = snakes.filter((s) => s.alive).sort((a, b) => b.length - a.length);
        if (order.length) {
          const cur = order.findIndex((s) => s.id === heroId);
          const base = cur < 0 ? 0 : cur;
          const ni =
            e.key === "]"
              ? (base + 1) % order.length
              : (base - 1 + order.length) % order.length;
          send("set_hero", order[ni].id);
        }
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (playing && e.code === "Space" && !isTyping(e)) {
        e.preventDefault();
        send("human_boost", false);
      }
    };
    // Boost is hold-to-activate; if the window loses focus (Cmd-Tab, click away)
    // while Space is held, keyup never fires — so release boost on blur/hide too,
    // otherwise it silently drains the snake with no key down.
    const releaseBoost = () => {
      if (playing) send("human_boost", false);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", releaseBoost);
    document.addEventListener("visibilitychange", releaseBoost);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", releaseBoost);
      document.removeEventListener("visibilitychange", releaseBoost);
    };
  }, [send, playing, session?.playing, showHelp, showPalette, showTour]);

  const linkLabel = {
    live: "● live",
    stale: "◐ stalled",
    reconnecting: "○ reconnecting…",
    offline: "○ offline",
  }[link];
  const linkClass = { live: "on", stale: "warn", reconnecting: "warn", offline: "off" }[link];

  return (
    <div className={"app" + (link === "stale" ? " stale" : "")}>
      <div className="topbar">
        <h1>
          snake-dqn <span className="dim">· live</span>
        </h1>
        <span className={"pill " + linkClass} title={`Stream: ${link}`}>
          {linkLabel}
        </span>
        {session && <span className="pill">{session.checkpoint ?? "no model"}</span>}
        <div className="spacer" />
        {/* the per-metric stats live in the telemetry band now; the header stays lean */}
        {playing && frame?.play ? (
          <>
            <span className="pill on">▶ playing</span>
            <span className="pill">score {frame.play.score}</span>
            <span className="pill">length {frame.play.length}</span>
          </>
        ) : (
          stats && (
            <>
              <span className="pill">frame {stats.frame.toLocaleString()}</span>
              {session && <span className="pill mode-pill">{session.mode}</span>}
            </>
          )
        )}
        <button
          className="pill pill-btn cmdk-pill"
          onClick={() => setShowPalette(true)}
          aria-label="Open command palette"
          title={`Command palette (${CMDK_LABEL})`}
        >
          <span className="cmdk-icon">⌘</span>K
        </button>
        <ViewMenu settings={view} update={updateView} />
        <button
          className="pill pill-btn"
          onClick={() => setShowHelp(true)}
          aria-label="Keyboard shortcuts"
          title="Keyboard shortcuts (?)"
        >
          ?
        </button>
      </div>

      <TelemetryBand history={history} />

      <div className="main">
        <GameCanvas
          frame={frame}
          onPickHero={(id) => send("set_hero", id)}
          showGhosts={showGhosts}
          view={view}
        />

        <div className="sidebar">
          <div
            className="tabs"
            role="tablist"
            aria-label="Panels"
            onKeyDown={(e) => {
              // roving arrow nav — disabled in Play so arrows keep steering
              if (playing) return;
              const i = TABS.findIndex((t) => t.id === tab);
              let ni = i;
              if (e.key === "ArrowRight") ni = (i + 1) % TABS.length;
              else if (e.key === "ArrowLeft") ni = (i - 1 + TABS.length) % TABS.length;
              else if (e.key === "Home") ni = 0;
              else if (e.key === "End") ni = TABS.length - 1;
              else return;
              e.preventDefault();
              setTab(TABS[ni].id);
              requestAnimationFrame(() =>
                document.getElementById(`tab-${TABS[ni].id}`)?.focus()
              );
            }}
          >
            {TABS.map((t) => (
              <button
                key={t.id}
                id={`tab-${t.id}`}
                role="tab"
                aria-selected={tab === t.id}
                aria-controls="tabpanel"
                tabIndex={tab === t.id ? 0 : -1}
                className={"tab" + (tab === t.id ? " active" : "")}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="tabpanel" role="tabpanel" id="tabpanel" aria-labelledby={`tab-${tab}`}>
            {!frame ? (
              <div className="panel">
                <div className="empty-state">
                  <div className="empty-glyph spin">◜</div>
                  <div className="empty-title">
                    {link === "reconnecting" || link === "offline"
                      ? "Reconnecting to the engine…"
                      : "Connecting to the engine…"}
                  </div>
                  <div className="muted" style={{ fontSize: 12 }}>
                    Streaming the live game over a WebSocket.
                  </div>
                </div>
              </div>
            ) : tab === "controls" && session ? (
              <Controls
                session={session}
                snakes={frame.snakes}
                send={send}
                ghosts={showGhosts}
                onToggleGhosts={() => setShowGhosts((g) => !g)}
              />
            ) : tab === "play" ? (
              <Play
                play={frame.play}
                send={send}
                speed={session?.speed ?? 12}
                error={session?.error ?? null}
              />
            ) : tab === "inspector" ? (
              <Inspector
                inspector={frame.inspector}
                tick={frame.frame}
                heroId={session?.hero_id ?? 0}
              />
            ) : tab === "raster" ? (
              <EgoRasterViewer
                raster={frame.hero_raster ?? null}
                obsSpec={frame.obs_spec}
                labels={view.labels}
              />
            ) : tab === "network" ? (
              <NetworkVisualizer netviz={frame.netviz} />
            ) : (
              <Dashboard checkpoint={session?.checkpoint ?? null} />
            )}
          </div>
        </div>
      </div>

      {showTour && <WelcomeTour onClose={() => setShowTour(false)} />}
      {showPalette && <CommandPalette commands={commands} onClose={() => setShowPalette(false)} />}
      {showHelp && (
        <HelpOverlay
          onClose={() => setShowHelp(false)}
          onReplayTour={() => {
            setShowHelp(false);
            replayTour();
          }}
        />
      )}
      {toast && <Toast message={toast.msg} tone={toast.tone} onClose={dismissToast} />}
    </div>
  );
}
