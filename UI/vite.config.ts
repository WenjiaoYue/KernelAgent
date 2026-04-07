import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    host: true,
    allowedHosts: 'all',
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
