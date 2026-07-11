// Transient status message (e.g. a model-load error, or a "copied ✓" confirm).
// Presentational only — App owns the message and its auto-dismiss timer.
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
    <div className={"toast toast-" + tone} role="status" aria-live="polite">
      <span>{message}</span>
      <button className="toast-x" onClick={onClose} aria-label="Dismiss">
        ✕
      </button>
    </div>
  );
}
