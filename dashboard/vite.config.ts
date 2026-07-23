/// <reference types="vitest/config" />
import solidPlugin from "vite-plugin-solid"
import { defineConfig } from "vitest/config"

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 8080,
    // Same-origin /api/v1 default works in dev by proxying to the local API
    // (make api-up). Override at runtime with ?api= as before.
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    target: "es2022",
    outDir: "dist",
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    deps: {
      optimizer: {
        web: {
          include: ["solid-js"],
        },
      },
    },
  },
})
