import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
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

const ControlsPanel = Controls;
const PlayPanel = Play;

function tabFromHash(): Tab | null {
  const h = window.location.hash.replace(/^#/, "");
  return TABS.some((t) => t.id === h) ? (h as Tab) : null;
}

// Restore the tab from the URL hash (deep link) or the last saved tab. The Play
// TAB is pure navigation — selecting it never touches the engine mode and never
// auto-starts a run, so restoring it (even from a shared link) is safe.
function initialTab(): Tab {
  const fromHash = tabFromHash();
  if (fromHash) return fromHash;
  const saved = localStorage.getItem("snake_tab");
  if (saved && TABS.some((t) => t.id === saved)) return saved as Tab;
  return "controls";
}

// Visually-hidden-but-announced style for the persistent live regions (an
// aria-live region must exist BEFORE its text changes to be reliably announced).
const SR_ONLY: CSSProperties = {
  position: "absolute",
  width: 1,
  height: 1,
  padding: 0,
  margin: -1,
  overflow: "hidden",
  clip: "rect(0 0 0 0)",
  whiteSpace: "nowrap",
  border: 0,
};

function compactNum(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}k`;
  return String(n);
}

export default function App() {
  const { frame, connected, link, history, notice, send } = useGameSocket();
  const [tab, setTab] = useState<Tab>(initialTab);
  const [showHelp, setShowHelp] = useState(false);
  const [showPalette, setShowPalette] = useState(false);
  const [showTour, setShowTour] = useState(() => !localStorage.getItem("snake_onboarded"));
  const [showGhosts, setShowGhosts] = useState(() => localStorage.getItem("snake_ghosts") !== "0");
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [toast, setToast] = useState<{ msg: string; tone: "error" | "ok" } | null>(null);
  // Mode change requested but not yet confirmed by a frame ("Switching…" UI).
  const [pendingMode, setPendingMode] = useState<string | null>(null);
  // Destructive-action confirm ("Training progress lives only in memory…").
  const [confirmGuard, setConfirmGuard] = useState<{ run: () => void } | null>(null);
  // Offer to retry a reward-contract-refused action as an explicit fine-tune.
  const [overrideOffer, setOverrideOffer] = useState<{ message: string } | null>(null);
  // The last destructive action issued, so an overridable refusal can be
  // re-sent with override_reward_contract once the user consents.
  const lastIntentRef = useRef<
    { action: "set_mode"; mode: string } | { action: "load_checkpoint"; name: string } | null
  >(null);
  const [view, updateView] = useViewSettings();
  const lastErrorRef = useRef<string | null>(null);
  // Bumped on every user-issued destructive action so the error-toast effect
  // re-runs even when the server reports the SAME error text as last time.
  const [errorAttempt, setErrorAttempt] = useState(0);
  // Latest snakes + hero for keyboard hero-cycling, without re-binding the
  // listener every frame.
  const heroCycleRef = useRef<{ snakes: SnakeDTO[]; heroId: number }>({ snakes: [], heroId: 0 });
  heroCycleRef.current = { snakes: frame?.snakes ?? [], heroId: frame?.session?.hero_id ?? 0 };
  const stats = frame?.stats;
  const session = frame?.session;
  const playing = session?.mode === "play";
  // Fresh in stable closures/effects without re-binding per frame:
  const runOverRef = useRef(false);
  runOverRef.current = !!frame?.play?.run_over;
  const runLiveRef = useRef(false);
  runLiveRef.current = !!(frame?.play?.run_started && !frame?.play?.run_over);
  const sessionPlayingRef = useRef(false);
  sessionPlayingRef.current = !!session?.playing;
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const statsFrameRef = useRef(0);
  statsFrameRef.current = stats?.frame ?? 0;
  const tabRef = useRef(tab);
  tabRef.current = tab;

  // Persist + mirror the active tab into the URL hash. replaceState keeps panels
  // linkable/bookmarkable WITHOUT history-entry spam, so the browser Back button
  // still leaves the app in one step instead of trapping the user in tab history.
  useEffect(() => {
    localStorage.setItem("snake_tab", tab);
    try {
      // NB: `history` in scope is the metric history — this is window.history.
      window.history.replaceState(null, "", `#${tab}`);
    } catch {
      /* sandboxed context — non-fatal */
    }
  }, [tab]);

  // Manual hash edits (or a pasted #tab link on an already-open page) select the
  // tab; an unknown hash is ignored. Never starts a run — tabs are navigation.
  useEffect(() => {
    const onHash = () => {
      const t = tabFromHash();
      if (t) setTab(t);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

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

  // Raster streaming is opt-in per connection: subscribe while the Raster tab is
  // active, unsubscribe when it leaves (the backend skips the heavy raster block
  // for unsubscribed clients).
  useEffect(() => {
    if (tab !== "raster") return;
    send("set_raster_stream", { on: true });
    return () => send("set_raster_stream", { on: false });
  }, [tab, send]);

  // On (re)connect, re-send ONLY the client's subscription intent. The mode is
  // NOT re-sent — the client adopts whatever the server broadcasts.
  const awaitingFirstFrameRef = useRef(true);
  const prevConnectedRef = useRef(false);
  useEffect(() => {
    if (connected && !prevConnectedRef.current) {
      awaitingFirstFrameRef.current = true;
      if (tabRef.current === "raster") send("set_raster_stream", { on: true });
    }
    prevConnectedRef.current = connected;
  }, [connected, send]);

  // First frame after a (re)connect: adopt the server's mode — clear any pending
  // "switching mode" UI (fixes the eternal "Switching to play mode…" spinner
  // after a backend restart), and if a play run is live server-side while we're
  // on another tab (e.g. after an F5 mid-run), surface it by selecting Play.
  useEffect(() => {
    if (!frame) return;
    if (awaitingFirstFrameRef.current) {
      awaitingFirstFrameRef.current = false;
      setPendingMode(null);
      if (frame.session?.mode === "play" && tabRef.current !== "play") setTab("play");
    } else if (pendingMode && frame.session?.mode === pendingMode) {
      setPendingMode(null);
    }
  }, [frame, pendingMode]);

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

  // Server-pushed {type:"error"|"info"} messages (engine crashes, "Saved …"
  // confirmations) surface as toasts — previously they were silently dropped.
  useEffect(() => {
    if (!notice) return;
    showToast(notice.message, notice.tone === "error" ? "error" : "ok", notice.tone === "error" ? 6000 : 4000);
  }, [notice, showToast]);

  // One-time warning if the server speaks a protocol version we don't know
  // (absent = pre-versioned backend, accepted silently).
  const protocolVersion = frame?.protocol_version;
  const protoWarnedRef = useRef(false);
  useEffect(() => {
    if (protocolVersion !== undefined && protocolVersion !== 2 && !protoWarnedRef.current) {
      protoWarnedRef.current = true;
      showToast(
        `Server protocol v${protocolVersion} — this page expects v2. Hard-refresh to update.`,
        "error",
        8000
      );
    }
  }, [protocolVersion, showToast]);

  // Destructive-action guard: in-memory training progress is destroyed by mode
  // changes, checkpoint loads and play-run starts. Confirm first (with a
  // save_weights escape hatch) when leaving train mode with progress.
  const guardDestructive = useCallback((run: () => void) => {
    const s = sessionRef.current;
    if (s?.mode === "train" && statsFrameRef.current > 0) setConfirmGuard({ run });
    else run();
  }, []);

  const changeMode = useCallback(
    (m: string) => {
      if (sessionRef.current?.mode === m) return; // server-side no-op too
      guardDestructive(() => {
        // Re-arm the error toast: retrying a failing switch must re-surface the
        // same failure text, not fail silently because the text didn't change.
        lastErrorRef.current = null;
        setErrorAttempt((n) => n + 1);
        lastIntentRef.current = { action: "set_mode", mode: m };
        setPendingMode(m);
        send("set_mode", m);
      });
    },
    [guardDestructive, send]
  );

  const loadCheckpoint = useCallback(
    (name: string) => {
      guardDestructive(() => {
        lastErrorRef.current = null;
        setErrorAttempt((n) => n + 1);
        lastIntentRef.current = { action: "load_checkpoint", name };
        send("load_checkpoint", name);
      });
    },
    [guardDestructive, send]
  );

  // Explicit start of a scored human run — the ONLY place (besides the palette
  // command that calls it) that enters play mode. The Play TAB never does.
  const startRun = useCallback(() => {
    setTab("play");
    if (sessionRef.current?.mode === "play") {
      send("new_game");
      return;
    }
    guardDestructive(() => {
      lastErrorRef.current = null;
      setErrorAttempt((n) => n + 1);
      setPendingMode("play");
      send("set_mode", "play");
    });
  }, [guardDestructive, send]);

  // Explicit end of the play session ("End run / back to watch" in the panel).
  const endRun = useCallback(() => {
    if (sessionRef.current?.mode !== "play") return;
    setPendingMode("watch");
    send("set_mode", "watch");
  }, [send]);

  // Command-palette actions, rebuilt only while open (keeps the hero/model lists
  // fresh without allocating a command list on every closed frame).
  const commands = useMemo<Command[]>(() => {
    if (!showPalette || !session || !frame) return [];
    const cmds: Command[] = [
      {
        id: "toggle-play",
        title: session.playing ? "Pause" : "Play",
        group: "Playback",
        hint: playing ? "P" : "Space",
        run: () => send(session.playing ? "pause" : "play"),
      },
      {
        id: "reset",
        title: "Reset game",
        group: "Playback",
        hint: playing ? undefined : "R",
        run: () => send("reset"),
      },
      {
        id: "mode-watch",
        title: "Watch mode",
        group: "Mode",
        active: session.mode === "watch",
        run: () => changeMode("watch"),
      },
      {
        id: "mode-train",
        title: "Train mode — learn live",
        group: "Mode",
        keywords: "learning",
        active: session.mode === "train",
        run: () => changeMode("train"),
      },
      {
        id: "mode-play",
        title: "Race the AI — start a play run",
        group: "Mode",
        keywords: "human versus play mode",
        active: session.mode === "play",
        run: startRun,
      },
      {
        id: "intent",
        title: showGhosts ? "Hide intent overlay" : "Show intent overlay",
        group: "View",
        hint: playing ? undefined : "O",
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
        run: () => loadCheckpoint(c.name),
      });
    }
    return cmds;
  }, [showPalette, session, frame, tab, checkpoints, showGhosts, playing, send, changeMode, startRun, loadCheckpoint, replayTour, copySummary]);

  // Surface session errors (bad checkpoint / mode swap) as a transient toast,
  // firing only on the transition to a new error (it persists across frames).
  // A failed mode swap also clears the pending "Switching…" UI.
  const sessionError = session?.error ?? null;
  const sessionErrorOverridable = !!session?.error_overridable;
  useEffect(() => {
    if (sessionError && sessionError !== lastErrorRef.current) {
      if (sessionErrorOverridable && lastIntentRef.current) {
        // The server says re-issuing with the reward override would work:
        // offer the explicit fine-tune instead of a dead-end error toast.
        setOverrideOffer({ message: sessionError });
      } else {
        // 8s, not a blink: this toast is usually the ONLY explanation of a
        // failed mode switch / checkpoint load, and the messages run long.
        showToast(sessionError, "error", 8000);
      }
      lastErrorRef.current = sessionError;
      setPendingMode(null);
    }
    if (!sessionError) {
      lastErrorRef.current = null;
      // The refusal the offer was about no longer exists (override succeeded,
      // or some other rebuild cleared it) — close a stale offer.
      setOverrideOffer((o) => (o ? null : o));
    }
  }, [sessionError, sessionErrorOverridable, errorAttempt, showToast]);

  // While an overlay (palette / help / tour) is up during play: force-release
  // boost (its Space keyup would be swallowed by the overlay's input), and
  // auto-pause a LIVE run so the snake doesn't sail into a wall while the user
  // reads — resuming on close only if we were the ones who paused it.
  const overlayOpen = showPalette || showHelp || showTour;
  const pausedByOverlayRef = useRef(false);
  useEffect(() => {
    if (!overlayOpen || !playing) return;
    send("human_boost", false);
    if (sessionPlayingRef.current && runLiveRef.current) {
      pausedByOverlayRef.current = true;
      send("pause");
    }
    return () => {
      if (pausedByOverlayRef.current) {
        pausedByOverlayRef.current = false;
        send("play");
      }
    };
  }, [overlayOpen, playing, send]);

  // Keyboard: in play mode, arrows/WASD steer (by physical key code), Space
  // boosts, P pauses, 1-5 set speed. Otherwise the classic Space = play/pause,
  // R = reset shortcuts apply. Ignored while typing.
  const pausedByHideRef = useRef(false);
  useEffect(() => {
    const isTyping = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
    };
    const isActivatable = (el: Element | null) => {
      const t = el?.tagName;
      return t === "BUTTON" || t === "A" || t === "INPUT" || t === "SELECT" || t === "TEXTAREA";
    };
    const speedPreset = (e: KeyboardEvent): boolean => {
      if (e.key >= "1" && e.key <= "5") {
        const p = SPEED_PRESETS[Number(e.key) - 1];
        if (p) {
          e.preventDefault();
          send("set_speed", p.fps);
          return true;
        }
      }
      return false;
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
      if (confirmGuard) {
        // the confirm dialog owns the keyboard; Escape cancels
        if (e.key === "Escape") {
          e.preventDefault();
          setConfirmGuard(null);
        }
        return;
      }
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
        // After a run ends, R or Enter starts a fresh game instantly — but Enter
        // must NOT hijack a focused button/link/input (e.g. Tab-to-Submit-Enter
        // must SUBMIT the score, not destroy it with a new game).
        if (runOverRef.current && (e.key === "r" || e.key === "R" || e.key === "Enter")) {
          if (e.key === "Enter" && isActivatable(document.activeElement)) return;
          e.preventDefault();
          send("new_game");
          return;
        }
        // P toggles pause during a run (Space is taken by boost).
        if (!e.repeat && (e.key === "p" || e.key === "P")) {
          e.preventDefault();
          send(sessionPlayingRef.current ? "pause" : "play");
          return;
        }
        // Speed presets stay available while playing (they don't steer).
        if (speedPreset(e)) return;
        const dir = resolveMoveKey(e.code, e.key);
        if (dir) {
          e.preventDefault();
          // held keys auto-repeat; the direction only needs to be sent once
          if (!e.repeat) send("human_input", dir);
        } else if (e.code === "Space") {
          // After the run, let Space activate a focused control (Submit button)
          // instead of being a dead boost key.
          if (runOverRef.current && isActivatable(document.activeElement)) return;
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
      } else if (speedPreset(e)) {
        return;
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
      // Releasing boost is idempotent and always safe — no isTyping guard, so a
      // Space keyup that lands in the palette's autofocused input still releases.
      if (playing && e.code === "Space") send("human_boost", false);
    };
    // Boost is hold-to-activate; if the window loses focus (Cmd-Tab, click away)
    // while Space is held, keyup never fires — so release boost on blur too,
    // otherwise it silently drains the snake with no key down.
    const releaseBoost = () => {
      if (playing) send("human_boost", false);
    };
    // Tab hidden mid-run: release boost AND pause the engine (the snake would
    // otherwise glide unsteered into a wall off-screen). Never auto-resumes —
    // the user resumes explicitly (P / the pause control).
    const onVisibility = () => {
      if (document.hidden) {
        if (playing) {
          send("human_boost", false);
          if (sessionPlayingRef.current && runLiveRef.current) {
            pausedByHideRef.current = true;
            send("pause");
          }
        }
      } else if (pausedByHideRef.current) {
        pausedByHideRef.current = false;
        showToast("Paused while you were away — press P to resume", "ok", 5000);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", releaseBoost);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", releaseBoost);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [send, playing, session?.playing, showHelp, showPalette, showTour, confirmGuard, showToast]);

  // Dynamic tab title: a backgrounded trainer/game should tell its status from
  // the tab strip alone (disconnected > play score > paused > train progress).
  useEffect(() => {
    let t: string;
    if (!connected) t = "○ disconnected · snake-dqn";
    else if (playing && frame?.play)
      t = frame.play.run_over
        ? `run over · score ${frame.play.score} · snake-dqn`
        : `▶ score ${frame.play.score} · snake-dqn`;
    else if (session && !session.playing) t = "⏸ paused · snake-dqn";
    else if (session?.mode === "train" && stats) t = `train · f ${compactNum(stats.frame)} · snake-dqn`;
    else if (stats) t = `watch · best ${stats.best_length} · snake-dqn`;
    else t = "snake-dqn · live";
    if (document.title !== t) document.title = t;
  }, [connected, playing, frame, session, stats]);

  // Stable callbacks so memoized panels don't re-render on identity churn.
  const pickHero = useCallback((id: number) => send("set_hero", id), [send]);
  const toggleGhosts = useCallback(() => setShowGhosts((g) => !g), []);

  const linkLabel = {
    live: "● live",
    stale: "◐ stalled",
    reconnecting: "○ reconnecting…",
    offline: "○ offline",
  }[link];
  const linkClass = { live: "on", stale: "warn", reconnecting: "warn", offline: "off" }[link];

  const confirmSave = useCallback(() => {
    if (!confirmGuard) return;
    send("save_weights");
    const g = confirmGuard;
    setConfirmGuard(null);
    g.run();
  }, [confirmGuard, send]);
  const confirmDiscard = useCallback(() => {
    if (!confirmGuard) return;
    const g = confirmGuard;
    setConfirmGuard(null);
    g.run();
  }, [confirmGuard]);
  const confirmCancel = useCallback(() => setConfirmGuard(null), []);

  // Reward-override offer: consent re-sends the refused action with the
  // validator's escape hatch armed (fine-tune under current rewards).
  const overrideAccept = useCallback(() => {
    const intent = lastIntentRef.current;
    setOverrideOffer(null);
    // Deliberately do NOT re-arm the error latch here: frames streamed while
    // the override build is in flight still carry the old overridable error,
    // and re-arming would immediately reopen this dialog. A genuine override
    // failure produces a DIFFERENT error text, which transitions on its own.
    if (!intent) return;
    if (intent.action === "set_mode") {
      setPendingMode(intent.mode);
      send("set_mode", { mode: intent.mode, override_reward_contract: true });
    } else {
      send("load_checkpoint", { name: intent.name, override_reward_contract: true });
    }
  }, [send]);
  const overrideCancel = useCallback(() => setOverrideOffer(null), []);

  return (
    <div className={"app" + (link === "stale" ? " stale" : "")}>
      {/* Persistent live regions: mounted from first render so screen readers
          track them; only their TEXT changes (a region that appears already
          populated is frequently not announced at all). */}
      <div role="status" aria-live="polite" style={SR_ONLY}>
        {toast && toast.tone !== "error" ? toast.msg : ""}
      </div>
      <div role="alert" aria-live="assertive" style={SR_ONLY}>
        {toast && toast.tone === "error" ? toast.msg : ""}
      </div>

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
          onPickHero={pickHero}
          showGhosts={showGhosts}
          view={view}
        />

        <div className="sidebar">
          <div
            className="tabs"
            role="tablist"
            aria-label="Panels"
            onKeyDown={(e) => {
              // Roving arrow nav. In Play mode arrows keep steering the snake, so
              // panel switching degrades to Tab + Enter (every tab button is
              // tabbable while playing — see tabIndex below).
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
                // Roving tabindex normally; while playing, arrows steer the
                // snake, so every tab stays reachable with the Tab key instead —
                // otherwise Play mode is a keyboard-only navigation trap.
                tabIndex={playing || tab === t.id ? 0 : -1}
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
              <ControlsPanel
                session={session}
                snakes={frame.snakes}
                send={send}
                ghosts={showGhosts}
                onToggleGhosts={toggleGhosts}
                connected={connected}
                onModeChange={changeMode}
                onLoadCheckpoint={loadCheckpoint}
                onCopySummary={copySummary}
              />
            ) : tab === "play" ? (
              <PlayPanel
                play={frame.play}
                send={send}
                speed={session?.speed ?? 12}
                error={session?.error ?? null}
                connected={connected}
                pendingMode={pendingMode}
                onStartRun={startRun}
                onEndRun={endRun}
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
      {confirmGuard && (
        <div className="overlay-backdrop" onClick={confirmCancel}>
          <div
            className="overlay-card"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="confirm-guard-title"
            style={{ maxWidth: 440 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="overlay-head">
              <h2 id="confirm-guard-title">Discard training progress?</h2>
            </div>
            <p className="muted" style={{ margin: "6px 0 14px", fontSize: 13 }}>
              Training progress lives only in memory. Save weights first?
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", flexWrap: "wrap" }}>
              <button className="btn primary" onClick={confirmSave} disabled={!connected}>
                Save &amp; continue
              </button>
              <button className="btn" onClick={confirmDiscard}>
                Discard
              </button>
              <button className="btn" onClick={confirmCancel} autoFocus>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      {overrideOffer && (
        <div className="overlay-backdrop" onClick={overrideCancel}>
          <div
            className="overlay-card"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="override-offer-title"
            style={{ maxWidth: 460 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="overlay-head">
              <h2 id="override-offer-title">Fine-tune across reward versions?</h2>
            </div>
            <p className="muted" style={{ margin: "6px 0 10px", fontSize: 13 }}>
              {overrideOffer.message}
            </p>
            <p className="muted" style={{ margin: "0 0 14px", fontSize: 13 }}>
              Continuing warm-starts the weights and trains them under the arena&rsquo;s
              current reward economics — a deliberate fine-tune, not a resume. Saved
              weights get stamped with the new contract.
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", flexWrap: "wrap" }}>
              <button className="btn primary" onClick={overrideAccept} disabled={!connected}>
                Fine-tune anyway
              </button>
              <button className="btn" onClick={overrideCancel} autoFocus>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      {toast && <Toast message={toast.msg} tone={toast.tone} onClose={dismissToast} />}
    </div>
  );
}
