import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import "./styles/theme.css";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { USE_MOCKS } from "./api/client";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5_000,
    },
  },
});

async function bootstrap() {
  if (USE_MOCKS) {
    try {
      const { startMocks } = await import("../mocks/browser");
      await startMocks();
    } catch (error) {
      // Service Worker registration fails in sandboxed and embedded browsers,
      // and over plain HTTP on a non-localhost host. Letting that reject here
      // meant createRoot() never ran and the whole console rendered blank --
      // a mock layer failing should degrade the app, not replace it with a
      // white page. Without mocks the client falls through to the real API and
      // the degraded-mode banner explains itself.
      console.warn("[mocks] disabled: Service Worker unavailable", error);
    }
  }

  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <ErrorBoundary>
            <App />
          </ErrorBoundary>
        </BrowserRouter>
      </QueryClientProvider>
    </StrictMode>,
  );
}

void bootstrap();
