import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ color: "white", background: "#0a0e1a", padding: "2rem", fontFamily: "monospace" }}>
          <h2 style={{ color: "#ff3b5c" }}>App Error</h2>
          <pre style={{ color: "#00d4ff", fontSize: "12px", whiteSpace: "pre-wrap" }}>
            {this.state.error?.toString()}
            {"\n"}
            {this.state.error?.stack}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary>
      <BrowserRouter basename="/ai-trading-dashboard">
        <App />
      </BrowserRouter>
    </ErrorBoundary>
  </React.StrictMode>
);