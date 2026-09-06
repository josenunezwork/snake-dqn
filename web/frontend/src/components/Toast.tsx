// Transient status message (e.g. a model-load error, or a "copied ✓" confirm).
// Presentational only — App owns the message, its auto-dismiss timer, AND the
// screen-reader announcement (persistent aria-live regions mounted in App).
// This visual card is deliberately NOT a live region: it mounts already
// populated, which screen readers frequently fail to announce.
export default function Toast({
  message,
  tone = "error",
  onClose,
}: {
  message: string;
  tone?: "error" | "ok";
  onClose: () => void;
}) {
  return (
    <div className={"toast toast-" + tone}>
      <span>{message}</span>
      <button className="toast-x" onClick={onClose} aria-label="Dismiss">
        ✕
      </button>
    </div>
  );
}
