import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API paths the same way nginx does in the container,
// to the ports docker-compose publishes.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api/rec": { target: "http://127.0.0.1:8001", rewrite: (p) => p.replace(/^\/api\/rec/, "") },
      "/api/explain": { target: "http://127.0.0.1:8002", rewrite: (p) => p.replace(/^\/api\/explain/, "") },
    },
  },
});
