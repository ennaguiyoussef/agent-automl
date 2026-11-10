import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// The API base URL is read at runtime from VITE_API_BASE_URL; in dev we also
// proxy /api and the SSE endpoint to the backend so cookies/CORS are a non-issue.
const API_TARGET = process.env.VITE_API_BASE_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
});
