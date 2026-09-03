import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";
import process from "node:process";

// Offline-first: no CDN, no external tile provider, no telemetry.
// Everything the venue demo needs is bundled or served by the local API.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
      "@contracts": fileURLToPath(new URL("./contracts", import.meta.url)),
      "@fixtures": fileURLToPath(new URL("./mocks/fixtures", import.meta.url)),
    },
  },
  server: { port: Number(process.env.PORT) || 5173 },
  build: {
    // Budget: < 500 KB gzipped JS excluding MapLibre. Split the heavy,
    // route-local dependencies so the demo path never pays for them.
    rollupOptions: {
      output: {
        manualChunks: {
          maplibre: ["maplibre-gl", "react-map-gl"],
          charts: ["recharts"],
        },
      },
    },
    chunkSizeWarningLimit: 900,
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
} as Parameters<typeof defineConfig>[0]);
