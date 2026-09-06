import { useEffect, useMemo, useRef, useState } from "react";
import { fuzzyFilter } from "../lib/fuzzy";

export interface Command {
  id: string;
  title: string;
  group: string;
  hint?: string; // keyboard shortcut or contextual note shown on the right
  keywords?: string; // extra search terms, not displayed
  active?: boolean; // renders a "current" dot (e.g. the live mode/speed/tab)
  run: () => void;
}

// A spotlight command bar (⌘K / Ctrl-K): fuzzy-search every action the app can
// take and run it with the keyboard. Reuses the modal backdrop styling. The list
// is supplied by the caller so this component stays a generic launcher.
export default function CommandPalette({
  commands,
  onClose,
}: {
  commands: Command[];
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [sel, setSel] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  // Captured during render: `autoFocus` on the input steals focus at commit time,
  // before any effect could snapshot who opened us.
  const [opener] = useState<HTMLElement | null>(() =>
    typeof document === "undefined" ? null : (document.activeElement as HTMLElement | null)
  );

  const results = useMemo(
    () => fuzzyFilter(commands, query, (c) => `${c.title} ${c.keywords ?? ""} ${c.group}`),
    [commands, query]
  );

  useEffect(() => setSel(0), [query]);

  // Hand focus back to whatever opened the palette (the opener can unmount if a
  // command swaps the view out from under it).
  useEffect(() => {
    return () => {
      if (opener?.isConnected) opener.focus();
    };
  }, [opener]);

  // keep the highlighted row in view
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${sel}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  const run = (c: Command | undefined) => {
    if (!c) return;
    c.run();
    onClose();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSel((s) => Math.min(s + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSel((s) => Math.max(s - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      run(results[sel]);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    } else if (e.key === "Home") {
      setSel(0);
    } else if (e.key === "End") {
      setSel(results.length - 1);
    } else if (e.key === "Tab") {
      // The input is the palette's only tab stop (rows are reached via
      // aria-activedescendant), so Tab must not walk out into the background
      // content that aria-modal declares inert.
      e.preventDefault();
    }
  };

  const activeOptionId = results[sel] ? `palette-opt-${sel}` : undefined;

  return (
    <div className="overlay-backdrop palette-backdrop" onClick={onClose}>
      <div
        className="palette"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          className="palette-input"
          autoFocus
          value={query}
          placeholder="Type a command… (play, train, 2×, inspect red, load champion)"
          aria-label="Search commands"
          role="combobox"
          aria-expanded={results.length > 0}
          aria-controls="palette-listbox"
          aria-autocomplete="list"
          aria-activedescendant={activeOptionId}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="palette-list" ref={listRef} id="palette-listbox" role="listbox">
          {results.length === 0 ? (
            <div className="palette-empty muted">No matching command</div>
          ) : (
            results.map((c, i) => (
              <button
                key={c.id}
                id={`palette-opt-${i}`}
                data-idx={i}
                role="option"
                aria-selected={i === sel}
                tabIndex={-1}
                className={"palette-row" + (i === sel ? " sel" : "")}
                onMouseMove={() => setSel(i)}
                onClick={() => run(c)}
              >
                {c.active && <span className="palette-dot" aria-label="current" />}
                <span className="palette-title">{c.title}</span>
                <span className="palette-group">{c.group}</span>
                {c.hint && <span className="palette-hint kbd">{c.hint}</span>}
              </button>
            ))
          )}
        </div>
        <div className="palette-foot muted">
          <span><span className="kbd">↑</span><span className="kbd">↓</span> navigate</span>
          <span><span className="kbd">↵</span> run</span>
          <span><span className="kbd">esc</span> close</span>
        </div>
      </div>
    </div>
  );
}
