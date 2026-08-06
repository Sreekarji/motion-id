import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// In production FastAPI serves this build, so relative API paths resolve on
// their own. Under `npm run dev` the app is on :5173 and the API on :8000, so
// these routes are proxied to keep the same relative URLs working in both.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/users': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
