import { setupWorker } from "msw/browser";
import { handlers } from "./handlers";

export const worker = setupWorker(...handlers);

/**
 * Mocks are the default. `VITE_USE_MOCKS=0` points the app at the live API —
 * that env var is the whole of "going live", per §9.
 */
export async function startMocks(): Promise<void> {
  await worker.start({
    onUnhandledRequest: "bypass",
    quiet: true,
    serviceWorker: { url: "/mockServiceWorker.js" },
  });
}
