import solidPlugin from "vite-plugin-solid";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "DITTO_");
  return {
    plugins: [solidPlugin()],
    server: {
      port: 8080,
      // Same-origin /api/v1 default works in dev by proxying to the local API
      // (make api-up). Override the target for preview QA without changing
      // browser CORS behavior.
      proxy: {
        "/api": env.DITTO_DASHBOARD_PROXY_TARGET || "http://localhost:8000",
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
  };
});
