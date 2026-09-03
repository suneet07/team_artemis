import { Suspense, lazy, useEffect } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { IdentityPlate } from "./components/IdentityPlate";
import { Skeleton, TipProvider } from "./components/primitives";
import { useWorkspace } from "./store/workspace";
import { LandingScreen } from "./features/landing/LandingScreen";
import { UploadScreen } from "./features/upload/UploadScreen";
import { PrepareScreen } from "./features/prepare/PrepareScreen";
import { WorkspaceScreen } from "./features/workspace/WorkspaceScreen";

// Lazy: the report route and the reliability diagram carry Recharts, which
// the demo path never needs (§12).
const ReportScreen = lazy(() =>
  import("./features/report/ReportScreen").then((m) => ({
    default: m.ReportScreen,
  })),
);
const SystemScreen = lazy(() =>
  import("./features/system/SystemScreen").then((m) => ({
    default: m.SystemScreen,
  })),
);

function RouteFallback() {
  return (
    <div className="mx-auto flex w-full max-w-[1180px] flex-col gap-3 px-6 py-8">
      <Skeleton className="h-7 w-64" />
      <Skeleton className="h-48 w-full" />
    </div>
  );
}

export function App() {
  const armature = useWorkspace((s) => s.armature);

  // The registration overlay is a document-level state so it can reveal both
  // the layout armature and the map's tiling partition at once.
  useEffect(() => {
    document.documentElement.dataset.armature = armature ? "on" : "off";
  }, [armature]);

  return (
    <TipProvider delayDuration={140} skipDelayDuration={300}>
      <div className="flex min-h-dvh flex-col bg-panel-0">
        <a
          href="#main"
          className="t-code sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:border focus:border-ink-0 focus:bg-panel-2 focus:px-3 focus:py-2"
        >
          Skip to content
        </a>
        <IdentityPlate />
        <main id="main" className="flex min-h-0 flex-1 flex-col">
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route path="/" element={<LandingScreen />} />
              <Route path="/upload" element={<UploadScreen />} />
              <Route path="/prepare/:bundleId" element={<PrepareScreen />} />
              <Route
                path="/workspace/:bundleId"
                element={<WorkspaceScreen />}
              />
              <Route
                path="/workspace/:bundleId/report/:queryId"
                element={<ReportScreen />}
              />
              <Route path="/system" element={<SystemScreen />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </TipProvider>
  );
}
