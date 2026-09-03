import { useEffect, useRef, useState } from "react";
import { resolveUrl } from "./client";

export type StreamTransport = "sse" | "polling" | "closed";

export interface StreamOptions {
  /** Named events to subscribe to. `message` is always included. */
  events: readonly string[];
  onEvent: (name: string, data: unknown, lastEventId: string) => void;
  /** Called after two failed EventSource attempts, to start the poll twin. */
  onFallback?: () => void;
  enabled?: boolean;
}

/**
 * Every stream in this API is a GET SSE endpoint, precisely so the native
 * `EventSource` can be used — which also gives us `Last-Event-ID` resumption
 * for free when the connection drops mid-stream.
 *
 * Per §3: after two failed attempts we stop retrying and hand over to the
 * polling twin (`GET /jobs/{job_id}` at 2 s), so a venue laptop that kills
 * long-lived connections still shows progress.
 */
export function useEventStream(
  url: string | null,
  { events, onEvent, onFallback, enabled = true }: StreamOptions,
): StreamTransport {
  const [transport, setTransport] = useState<StreamTransport>("closed");
  const onEventRef = useRef(onEvent);
  const onFallbackRef = useRef(onFallback);
  onEventRef.current = onEvent;
  onFallbackRef.current = onFallback;

  const eventsKey = events.join(",");

  useEffect(() => {
    if (!url || !enabled) {
      setTransport("closed");
      return;
    }

    let source: EventSource | null = null;
    let failures = 0;
    let disposed = false;
    let retryTimer: number | undefined;

    const open = () => {
      if (disposed) return;
      source = new EventSource(resolveUrl(url));
      setTransport("sse");

      const handle = (name: string) => (raw: MessageEvent<string>) => {
        failures = 0;
        let parsed: unknown = raw.data;
        try {
          parsed = JSON.parse(raw.data);
        } catch {
          /* a bare comment or heartbeat — pass the text through */
        }
        onEventRef.current(name, parsed, raw.lastEventId);
      };

      for (const name of eventsKey.split(",").filter(Boolean)) {
        source.addEventListener(name, handle(name) as EventListener);
      }
      source.addEventListener("message", handle("message") as EventListener);

      source.onerror = () => {
        source?.close();
        source = null;
        if (disposed) return;
        failures += 1;
        if (failures >= 2) {
          setTransport("polling");
          onFallbackRef.current?.();
          return;
        }
        retryTimer = window.setTimeout(open, 700);
      };
    };

    open();

    return () => {
      disposed = true;
      window.clearTimeout(retryTimer);
      source?.close();
      setTransport("closed");
    };
  }, [url, enabled, eventsKey]);

  return transport;
}

export const PREP_EVENTS = [
  "progress",
  "stage_complete",
  "done",
  "error",
] as const;

export const QUERY_EVENTS = [
  "accepted",
  "router",
  "validator",
  "plan",
  "step_started",
  "step_completed",
  "evidence",
  "agreement",
  "token",
  "fusion",
  "done",
  "error",
] as const;
