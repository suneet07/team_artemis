/**
 * Scripted SSE replay.
 *
 * Every stream in the contract is a GET `text/event-stream` with named
 * events and an `id:` on each, so the real `EventSource` — including its
 * `Last-Event-ID` resumption — is exercised in mock mode exactly as it will
 * be against the backend.
 *
 * Real timings are compressed so a demo is watchable: preparation is about
 * four minutes in life and about twenty seconds here; a query is about
 * twelve seconds in life and about four here. `VITE_MOCK_SPEED` scales it.
 */

const SPEED = Number(import.meta.env?.VITE_MOCK_SPEED ?? 1) || 1;

export interface ScriptedEvent {
  /** Milliseconds to wait before emitting, after the previous event. */
  after: number;
  event: string;
  data: unknown;
}

const encoder = new TextEncoder();

function frame(id: number, event: string, data: unknown): Uint8Array {
  return encoder.encode(
    `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
  );
}

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

export function sseResponse(
  script: ScriptedEvent[],
  options: { resumeFrom?: number } = {},
): Response {
  const resumeFrom = options.resumeFrom ?? 0;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      // A comment heartbeat every 15 s in life; here it just proves the
      // client tolerates non-event frames.
      controller.enqueue(encoder.encode(": stream open\n\n"));

      let heartbeat = 0;
      try {
        for (let index = 0; index < script.length; index += 1) {
          const step = script[index];
          const id = index + 1;

          // Resumption: replay is skipped for frames the client already has.
          if (id <= resumeFrom) continue;

          const wait = Math.max(0, step.after / SPEED);
          if (wait > 4000) {
            // Keep the connection provably alive across the long waits.
            let remaining = wait;
            while (remaining > 4000) {
              await sleep(4000);
              remaining -= 4000;
              heartbeat += 1;
              controller.enqueue(encoder.encode(`: ping ${heartbeat}\n\n`));
            }
            await sleep(remaining);
          } else {
            await sleep(wait);
          }

          controller.enqueue(frame(id, step.event, step.data));
        }
      } catch {
        /* the consumer disconnected — nothing to clean up */
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}

export function parseLastEventId(request: Request): number {
  const header = request.headers.get("Last-Event-ID");
  const parsed = header ? Number.parseInt(header, 10) : 0;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

/** Total scripted wall-clock, used to report a plausible `elapsed_ms`. */
export function scriptDuration(script: ScriptedEvent[]): number {
  return script.reduce((total, step) => total + step.after, 0);
}
