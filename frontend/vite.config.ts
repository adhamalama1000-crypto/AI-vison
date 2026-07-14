import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built assets are served by the FastAPI backend under /app/ (see app.py).
// The dev server proxies API + media + streams to the backend on :8090.
export default defineConfig({
  base: "/app/",
  plugins: [react()],
  build: {
    outDir: "../rtsp_backend/web",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8090",
      "/cameras": "http://127.0.0.1:8090",
      "/snapshot": "http://127.0.0.1:8090",
      "/stream": "http://127.0.0.1:8090",
      "/health": "http://127.0.0.1:8090",
      "/active-camera": "http://127.0.0.1:8090",
      "/ws": { target: "ws://127.0.0.1:8090", ws: true },
    },
  },
});
