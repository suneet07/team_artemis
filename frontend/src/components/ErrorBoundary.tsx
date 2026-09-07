import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * The console had no error boundary, which is why one small bug read as a
 * catastrophe: React unmounts the *entire* tree on an uncaught render error, so
 * a null dereference inside a `setTransform` updater turned the whole screen
 * blank. Nothing on screen, nothing in the UI to say what happened.
 *
 * A demo cannot afford that failure mode. This keeps the shell alive, names the
 * error, and offers a way back, so the worst case is one broken panel rather
 * than a white page in front of a judge.
 */
interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Kept: the stack is the only record of what happened, and swallowing it
    // silently is how a reproducible bug becomes an unreproducible one.
    console.error("Unhandled render error", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="m-sheet mx-auto my-8 max-w-2xl">
        <div className="border-b border-window-2 px-4 py-3">
          <span className="t-code-sm text-caution">RENDER ERROR</span>
        </div>
        <div className="space-y-3 px-4 py-4">
          <p className="text-ink-1">
            This panel failed to render. The rest of the console is still
            running, and nothing that was measured has been altered.
          </p>
          <pre className="t-data overflow-x-auto border border-window-2 bg-window-0/60 p-3 text-[11px] text-ink-2">
            {error.message}
          </pre>
          <button
            type="button"
            className="t-code-sm border border-window-2 px-3 py-1.5 text-ink-1 hover:bg-window-1"
            onClick={() => this.setState({ error: null })}
          >
            TRY AGAIN
          </button>
        </div>
      </div>
    );
  }
}
