import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" so the built dist works when served from any path.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
