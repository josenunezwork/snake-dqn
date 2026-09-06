import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

// Last-resort error boundary: without one, a single render-time exception (e.g.
// an unexpected frame shape from the backend) unmounts the whole root into a
// permanent white screen. Show the error and a reload escape hatch instead.
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("snake-dqn UI crashed:", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
        <div
          role="alert"
          style={{ maxWidth: 480, textAlign: "center", fontFamily: "system-ui, sans-serif" }}
        >
          <h1 style={{ fontSize: 18, marginBottom: 8 }}>Something broke in the UI</h1>
          <p
            style={{
              opacity: 0.75,
              fontSize: 13,
              marginBottom: 16,
              overflowWrap: "anywhere",
            }}
          >
            {String(this.state.error.message || this.state.error)}
          </p>
          <button
            className="btn"
            style={{ cursor: "pointer", padding: "8px 16px" }}
            onClick={() => location.reload()}
          >
            Reload the app
          </button>
        </div>
      </div>
    );
  }
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
